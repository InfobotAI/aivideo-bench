"""Strict MCP client primitives for official AIVideo Bench execution.

The benchmark must not approximate the tool surface it claims to evaluate.
This module discovers tools from an MCP server, validates and commits the exact
model-visible schema, and executes tool calls through Streamable HTTP. It has
no dependency on the MCP Python SDK so the benchmark's verification path stays
small and auditable.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol
from urllib.parse import urlsplit

from .mcp_credential_commitment import mcp_credential_material_sha256
from .standard import canonical_bytes


MCP_CLIENT_VERSION = "1"
MCP_PROTOCOL_VERSION = "2025-11-25"
SUPPORTED_MCP_PROTOCOL_VERSIONS = frozenset({"2025-11-25", "2025-06-18"})
TOOL_SURFACE_SCHEMA = "aivideo-bench-mcp-tool-surface-v1"
MCP_RUNTIME_ATTESTATION_SCHEMA = "aivideo-bench-mcp-runtime-attestation-v3"
MCP_TRANSPORT_PROFILE_STATELESS = "streamable_http_stateless"
MCP_TRANSPORT_PROFILE_SERVER_SESSION = "streamable_http_server_session"
MCP_TRANSPORT_PROFILES = frozenset(
    {
        MCP_TRANSPORT_PROFILE_STATELESS,
        MCP_TRANSPORT_PROFILE_SERVER_SESSION,
    }
)
MCP_RUNTIME_ATTESTATION_KEYS = frozenset(
    {
        "schema_version",
        "role",
        "factory_identity_sha256",
        "factory_configuration_sha256",
        "client_identity_sha256",
        "credential_id_sha256",
        "credential_material_sha256",
        "transport_profile",
        "protocol_version",
        "server_info_sha256",
        "tool_surface_sha256",
        "initialized",
        "surface_recheck_passed",
        "llm_inference_calls",
        "media_processing_calls",
        "tool_calls",
        "attestation_sha256",
    }
)
MCP_RUNTIME_ROLES = frozenset({"candidate", "observer"})
_MODEL_TOOL_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_TOOL_KEYS = frozenset(
    {
        "name",
        "title",
        "description",
        "inputSchema",
        "outputSchema",
        "annotations",
        "icons",
        "execution",
        "_meta",
    }
)
_SERVER_INFO_KEYS = frozenset(
    {
        "name",
        "title",
        "version",
        "description",
        "icons",
        "websiteUrl",
    }
)


class MCPProtocolError(RuntimeError):
    """The remote endpoint violated the negotiated MCP contract."""


class MCPToolSurfaceDriftError(RuntimeError):
    """The live MCP tool surface differs from the frozen benchmark surface."""


class HTTPTransport(Protocol):
    """One-attempt HTTP boundary used to make retry behavior testable."""

    max_attempts: int

    def request(
        self,
        *,
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes | None,
        timeout_seconds: float,
        max_response_bytes: int,
    ) -> tuple[int, Mapping[str, str], bytes]: ...


class UrllibSingleAttemptTransport:
    """Minimal HTTP transport with no implicit application-level retries."""

    max_attempts = 1

    class _NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            del req, fp, code, msg, headers, newurl
            return None

    def request(
        self,
        *,
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes | None,
        timeout_seconds: float,
        max_response_bytes: int,
    ) -> tuple[int, Mapping[str, str], bytes]:
        request = urllib.request.Request(
            url=url,
            data=body,
            headers=dict(headers),
            method=method,
        )
        try:
            opener = urllib.request.build_opener(self._NoRedirect())
            response = opener.open(request, timeout=timeout_seconds)
        except urllib.error.HTTPError as error:
            response_body = error.read(max_response_bytes + 1)
            if len(response_body) > max_response_bytes:
                raise MCPProtocolError("MCP HTTP response exceeds the byte limit")
            return int(error.code), dict(error.headers.items()), response_body
        with response:
            response_body = response.read(max_response_bytes + 1)
            if len(response_body) > max_response_bytes:
                raise MCPProtocolError("MCP HTTP response exceeds the byte limit")
            return int(response.status), dict(response.headers.items()), response_body


@dataclass(frozen=True)
class StreamableHTTPConfig:
    """Secret-bearing connection config; public identity never includes values."""

    endpoint: str
    credential_id: str
    headers: tuple[tuple[str, str], ...] = ()
    timeout_seconds: float = 300.0
    max_tool_pages: int = 32
    max_response_bytes: int = 8 * 1024 * 1024
    max_total_response_bytes: int = 32 * 1024 * 1024
    max_tool_result_bytes: int = 2 * 1024 * 1024
    allow_insecure_loopback: bool = False


def _implementation_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _safe_endpoint(endpoint: str, *, allow_insecure_loopback: bool) -> str:
    try:
        parsed = urlsplit(endpoint)
        parsed.port
    except ValueError as error:
        raise ValueError("MCP endpoint is malformed") from error
    if parsed.username or parsed.password:
        raise ValueError("MCP endpoint must not contain user information")
    if parsed.query or parsed.fragment:
        raise ValueError("MCP endpoint must not contain a query or fragment")
    if not parsed.hostname or not parsed.path:
        raise ValueError("MCP endpoint must contain a host and path")
    if parsed.scheme != "https":
        loopback = False
        try:
            loopback = ipaddress.ip_address(parsed.hostname).is_loopback
        except ValueError:
            loopback = parsed.hostname.casefold() == "localhost"
        if not (allow_insecure_loopback and parsed.scheme == "http" and loopback):
            raise ValueError("official MCP endpoints must use HTTPS")
    return endpoint


def _validated_headers(headers: tuple[tuple[str, str], ...]) -> dict[str, str]:
    result: dict[str, str] = {}
    protected = {
        "accept",
        "content-type",
        "mcp-protocol-version",
        "mcp-session-id",
    }
    for raw_name, raw_value in headers:
        name = str(raw_name).strip()
        value = str(raw_value).strip()
        if not name or not value or "\n" in name + value or "\r" in name + value:
            raise ValueError("MCP header names and values must be non-empty single lines")
        folded = name.casefold()
        if folded in protected:
            raise ValueError(f"MCP connection config cannot override {name}")
        if folded in {existing.casefold() for existing in result}:
            raise ValueError("MCP connection headers must be unique case-insensitively")
        result[name] = value
    return result


def validate_streamable_http_config(config: Any) -> list[str]:
    if not isinstance(config, StreamableHTTPConfig):
        return ["MCP config has the wrong type"]
    errors: list[str] = []
    try:
        _safe_endpoint(
            config.endpoint,
            allow_insecure_loopback=config.allow_insecure_loopback,
        )
    except ValueError as error:
        errors.append(str(error))
    try:
        _validated_headers(config.headers)
    except ValueError as error:
        errors.append(str(error))
    if not isinstance(config.credential_id, str) or len(config.credential_id.strip()) < 8:
        errors.append("MCP credential_id must be a non-secret stable identifier")
    if (
        isinstance(config.timeout_seconds, bool)
        or not isinstance(config.timeout_seconds, (int, float))
        or not 0 < float(config.timeout_seconds) <= 1800
    ):
        errors.append("MCP timeout_seconds must be in (0,1800]")
    if (
        isinstance(config.max_tool_pages, bool)
        or not isinstance(config.max_tool_pages, int)
        or not 1 <= config.max_tool_pages <= 100
    ):
        errors.append("MCP max_tool_pages must be in [1,100]")
    for key, upper in (
        ("max_response_bytes", 64 * 1024 * 1024),
        ("max_total_response_bytes", 256 * 1024 * 1024),
        ("max_tool_result_bytes", 16 * 1024 * 1024),
    ):
        value = getattr(config, key)
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 1024 <= value <= upper
        ):
            errors.append(f"MCP {key} must be an integer in [1024,{upper}]")
    if config.max_total_response_bytes < config.max_response_bytes:
        errors.append("MCP total response limit must cover at least one response")
    if config.max_tool_result_bytes > config.max_response_bytes:
        errors.append("MCP tool-result limit cannot exceed the response limit")
    return errors


def _canonical_tool(tool: Any) -> dict[str, Any]:
    if not isinstance(tool, dict):
        raise MCPProtocolError("MCP tools/list returned a non-object tool")
    unknown = set(tool) - _TOOL_KEYS
    if unknown:
        raise MCPProtocolError(
            "MCP tool contains unsupported fields: " + ", ".join(sorted(unknown))
        )
    name = tool.get("name")
    if not isinstance(name, str) or not _MODEL_TOOL_NAME.fullmatch(name):
        raise MCPProtocolError(
            "MCP tool name is not compatible with the model function interface"
        )
    description = tool.get("description", "")
    if description is None:
        description = ""
    if not isinstance(description, str):
        raise MCPProtocolError(f"MCP tool {name} description must be text")
    input_schema = tool.get("inputSchema")
    if not isinstance(input_schema, dict) or input_schema.get("type") != "object":
        raise MCPProtocolError(f"MCP tool {name} inputSchema must be an object schema")
    try:
        json.dumps(tool, ensure_ascii=True, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise MCPProtocolError(f"MCP tool {name} is not canonical JSON") from error
    result = {
        "name": name,
        "description": description,
        "inputSchema": input_schema,
    }
    for key in (
        "title",
        "outputSchema",
        "annotations",
        "icons",
        "execution",
        "_meta",
    ):
        if key in tool:
            result[key] = tool[key]
    return result


def build_tool_surface(
    *,
    protocol_version: str,
    server_info: Any,
    tools: list[Any],
) -> dict[str, Any]:
    """Normalize an MCP discovery result and bind ordering-independent identity."""

    if protocol_version not in SUPPORTED_MCP_PROTOCOL_VERSIONS:
        raise MCPProtocolError(
            f"unsupported negotiated MCP protocol version: {protocol_version}"
        )
    if not isinstance(server_info, dict):
        raise MCPProtocolError("MCP initialize result has no serverInfo object")
    unknown_server_info = set(server_info) - _SERVER_INFO_KEYS
    if unknown_server_info:
        raise MCPProtocolError(
            "MCP serverInfo contains unsupported fields: "
            + ", ".join(sorted(unknown_server_info))
        )
    if not isinstance(server_info.get("name"), str) or not server_info["name"].strip():
        raise MCPProtocolError("MCP serverInfo.name must be non-empty")
    if not isinstance(server_info.get("version"), str) or not server_info[
        "version"
    ].strip():
        raise MCPProtocolError("MCP serverInfo.version must be non-empty")
    normalized = [_canonical_tool(tool) for tool in tools]
    names = [tool["name"] for tool in normalized]
    if len(names) != len(set(names)):
        raise MCPProtocolError("MCP tools/list returned duplicate tool names")
    normalized.sort(key=lambda tool: tool["name"])
    safe_server_info = {
        key: server_info[key]
        for key in (
            "name",
            "title",
            "version",
            "description",
            "icons",
            "websiteUrl",
        )
        if key in server_info
    }
    surface: dict[str, Any] = {
        "schema_version": TOOL_SURFACE_SCHEMA,
        "protocol_version": protocol_version,
        "server_info": safe_server_info,
        "tools": normalized,
    }
    surface["tool_surface_sha256"] = _sha256(surface)
    return surface


def validate_tool_surface(surface: Any) -> list[str]:
    if not isinstance(surface, dict):
        return ["MCP tool surface must be an object"]
    if set(surface) != {
        "schema_version",
        "protocol_version",
        "server_info",
        "tools",
        "tool_surface_sha256",
    }:
        return ["MCP tool surface keys do not match the canonical schema"]
    try:
        rebuilt = build_tool_surface(
            protocol_version=surface["protocol_version"],
            server_info=surface["server_info"],
            tools=surface["tools"],
        )
    except (KeyError, MCPProtocolError, TypeError, ValueError) as error:
        return [str(error)]
    if rebuilt != surface:
        return ["MCP tool surface is not canonical or its commitment is invalid"]
    return []


def openrouter_tools(surface: dict[str, Any]) -> list[dict[str, Any]]:
    errors = validate_tool_surface(surface)
    if errors:
        raise ValueError("invalid MCP tool surface: " + "; ".join(errors))
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["inputSchema"],
            },
        }
        for tool in surface["tools"]
    ]


def tool_result_text(result: Any) -> str:
    """Losslessly expose an MCP CallToolResult to the model as canonical JSON."""

    if not isinstance(result, dict):
        raise MCPProtocolError("MCP tools/call result must be an object")
    content = result.get("content")
    if not isinstance(content, list):
        raise MCPProtocolError("MCP tools/call result content must be a list")
    if "isError" in result and not isinstance(result["isError"], bool):
        raise MCPProtocolError("MCP tools/call isError must be boolean")
    try:
        return json.dumps(
            result,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise MCPProtocolError("MCP tools/call result is not canonical JSON") from error


def _parse_sse(body: bytes) -> list[dict[str, Any]]:
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError as error:
        raise MCPProtocolError("MCP SSE response is not UTF-8") from error
    messages: list[dict[str, Any]] = []
    data_lines: list[str] = []
    for line in text.splitlines() + [""]:
        if not line:
            if data_lines:
                try:
                    value = json.loads("\n".join(data_lines))
                except json.JSONDecodeError as error:
                    raise MCPProtocolError("MCP SSE data is not JSON") from error
                if not isinstance(value, dict):
                    raise MCPProtocolError("MCP SSE message must be an object")
                messages.append(value)
                data_lines = []
            continue
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
    return messages


class StreamableHTTPMCPClient:
    """Synchronous, single-attempt MCP 2025-06-18 Streamable HTTP client."""

    def __init__(
        self,
        config: StreamableHTTPConfig,
        *,
        transport: HTTPTransport | None = None,
    ) -> None:
        errors = validate_streamable_http_config(config)
        if errors:
            raise ValueError("invalid MCP config: " + "; ".join(errors))
        self.config = config
        self.transport = transport or UrllibSingleAttemptTransport()
        if getattr(self.transport, "max_attempts", None) != 1:
            raise ValueError("official MCP transport must make exactly one HTTP attempt")
        self._connection_headers = _validated_headers(config.headers)
        self._next_id = 1
        self._session_id: str | None = None
        self._transport_profile: str | None = None
        self._protocol_version: str | None = None
        self._server_info: dict[str, Any] | None = None
        self._capabilities: dict[str, Any] | None = None
        self._tool_surface: dict[str, Any] | None = None
        self._closed = False
        self._response_bytes = 0
        self.transcript: list[dict[str, Any]] = []
        parsed = urlsplit(config.endpoint)
        safe_endpoint = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        self.identity = {
            "client_id": "aivideo-bench-streamable-http-mcp",
            "client_version": MCP_CLIENT_VERSION,
            "implementation_sha256": _implementation_sha256(),
            "configuration_sha256": _sha256(
                {
                    "endpoint": safe_endpoint,
                    "header_names": sorted(
                        name.casefold() for name in self._connection_headers
                    ),
                    "credential_id": config.credential_id,
                    "timeout_seconds": float(config.timeout_seconds),
                    "max_tool_pages": config.max_tool_pages,
                    "max_response_bytes": config.max_response_bytes,
                    "max_total_response_bytes": config.max_total_response_bytes,
                    "max_tool_result_bytes": config.max_tool_result_bytes,
                    "requested_protocol_version": MCP_PROTOCOL_VERSION,
                }
            ),
            "credential_material_sha256": mcp_credential_material_sha256(
                config.headers,
                required=False,
            ),
        }

    @property
    def tool_surface(self) -> dict[str, Any]:
        if self._tool_surface is None:
            raise RuntimeError("MCP client is not initialized")
        return json.loads(json.dumps(self._tool_surface))

    @property
    def transport_profile(self) -> str:
        """Return the server-session behavior observed during initialization.

        This is trusted-client evidence, not a server signature.  The value is
        frozen after initialization so a server cannot silently switch transport
        semantics during a benchmark cell.
        """

        if self._transport_profile is None:
            raise RuntimeError("MCP client transport profile is not initialized")
        return self._transport_profile

    def _headers(self) -> dict[str, str]:
        headers = dict(self._connection_headers)
        headers.update(
            {
                "Accept": "application/json, text/event-stream",
                "Content-Type": "application/json",
            }
        )
        if self._protocol_version is not None:
            headers["MCP-Protocol-Version"] = self._protocol_version
        if self._session_id is not None:
            headers["Mcp-Session-Id"] = self._session_id
        return headers

    @staticmethod
    def _header(headers: Mapping[str, str], name: str) -> str | None:
        for key, value in headers.items():
            if key.casefold() == name.casefold():
                return str(value)
        return None

    def _post(
        self,
        payload: dict[str, Any],
        *,
        expect_response: bool,
    ) -> dict[str, Any] | None:
        if self._closed:
            raise RuntimeError("MCP client is closed")
        if self.transport.max_attempts != 1:
            raise RuntimeError("MCP transport retry policy changed after initialization")
        request_bytes = (
            json.dumps(
                payload,
                ensure_ascii=True,
                sort_keys=False,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode()
        status, response_headers, body = self.transport.request(
            method="POST",
            url=self.config.endpoint,
            headers=self._headers(),
            body=request_bytes,
            timeout_seconds=float(self.config.timeout_seconds),
            max_response_bytes=self.config.max_response_bytes,
        )
        self._response_bytes += len(body)
        if self._response_bytes > self.config.max_total_response_bytes:
            raise MCPProtocolError("MCP cell responses exceed the total byte limit")
        self.transcript.append(
            {
                "method": payload.get("method"),
                "request_id": payload.get("id"),
                "request_sha256": hashlib.sha256(request_bytes).hexdigest(),
                "http_status": int(status),
                "response_sha256": hashlib.sha256(body).hexdigest(),
            }
        )
        if not 200 <= int(status) < 300:
            raise MCPProtocolError(f"MCP HTTP request failed with status {status}")
        returned_session = self._header(response_headers, "Mcp-Session-Id")
        if returned_session is not None:
            if not returned_session.isascii() or not all(
                0x21 <= ord(character) <= 0x7E for character in returned_session
            ):
                raise MCPProtocolError("MCP session ID contains invalid characters")
            if self._transport_profile == MCP_TRANSPORT_PROFILE_STATELESS:
                raise MCPProtocolError(
                    "MCP server introduced a session after stateless initialization"
                )
            if self._session_id is not None and returned_session != self._session_id:
                raise MCPProtocolError("MCP server changed session ID")
            self._session_id = returned_session
        if not expect_response:
            if body.strip():
                content_type = self._header(response_headers, "Content-Type") or ""
                if "application/json" in content_type:
                    try:
                        value = json.loads(body)
                    except json.JSONDecodeError as error:
                        raise MCPProtocolError("MCP notification response is invalid") from error
                    if value not in ({}, None):
                        raise MCPProtocolError(
                            "MCP notification unexpectedly returned a JSON-RPC message"
                        )
            return None
        content_type = self._header(response_headers, "Content-Type") or ""
        if "text/event-stream" in content_type:
            messages = _parse_sse(body)
        else:
            try:
                decoded = json.loads(body)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise MCPProtocolError("MCP response is not valid JSON") from error
            messages = [decoded] if isinstance(decoded, dict) else []
        expected_id = payload.get("id")
        matches = [message for message in messages if message.get("id") == expected_id]
        if len(matches) != 1:
            raise MCPProtocolError("MCP response did not contain exactly one matching ID")
        response = matches[0]
        if response.get("jsonrpc") != "2.0":
            raise MCPProtocolError("MCP response has an invalid JSON-RPC version")
        if "error" in response:
            error = response["error"]
            code = error.get("code") if isinstance(error, dict) else None
            raise MCPProtocolError(f"MCP JSON-RPC error {code}")
        result = response.get("result")
        if not isinstance(result, dict):
            raise MCPProtocolError("MCP response has no result object")
        return result

    def _request(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        request_id = self._next_id
        self._next_id += 1
        payload: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
        }
        if params is not None:
            payload["params"] = params
        result = self._post(payload, expect_response=True)
        assert result is not None
        return result

    def initialize(self) -> dict[str, Any]:
        if self._server_info is not None:
            raise RuntimeError("MCP client is already initialized")
        result = self._request(
            "initialize",
            {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {
                    "name": "aivideo-bench",
                    "version": MCP_CLIENT_VERSION,
                },
            },
        )
        protocol_version = result.get("protocolVersion")
        if protocol_version not in SUPPORTED_MCP_PROTOCOL_VERSIONS:
            raise MCPProtocolError(
                f"unsupported negotiated MCP protocol version: {protocol_version}"
            )
        capabilities = result.get("capabilities")
        if not isinstance(capabilities, dict) or not isinstance(
            capabilities.get("tools"), dict
        ):
            raise MCPProtocolError("MCP server did not declare the tools capability")
        server_info = result.get("serverInfo")
        if not isinstance(server_info, dict):
            raise MCPProtocolError("MCP server did not provide serverInfo")
        self._protocol_version = protocol_version
        self._server_info = server_info
        self._capabilities = capabilities
        self._post(
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            expect_response=False,
        )
        self._tool_surface = self._discover_tool_surface()
        self._transport_profile = (
            MCP_TRANSPORT_PROFILE_SERVER_SESSION
            if self._session_id is not None
            else MCP_TRANSPORT_PROFILE_STATELESS
        )
        return self.tool_surface

    def _discover_tool_surface(self) -> dict[str, Any]:
        if self._server_info is None or self._protocol_version is None:
            raise RuntimeError("MCP client must be initialized before tools/list")
        tools: list[Any] = []
        cursor: str | None = None
        seen_cursors: set[str] = set()
        for _page in range(self.config.max_tool_pages):
            params = {"cursor": cursor} if cursor is not None else None
            result = self._request("tools/list", params)
            page_tools = result.get("tools")
            if not isinstance(page_tools, list):
                raise MCPProtocolError("MCP tools/list result has no tools list")
            tools.extend(page_tools)
            next_cursor = result.get("nextCursor")
            if next_cursor is None:
                break
            if not isinstance(next_cursor, str) or not next_cursor:
                raise MCPProtocolError("MCP tools/list returned an invalid cursor")
            if next_cursor in seen_cursors:
                raise MCPProtocolError("MCP tools/list pagination cursor repeated")
            seen_cursors.add(next_cursor)
            cursor = next_cursor
        else:
            raise MCPProtocolError("MCP tools/list exceeded the page limit")
        return build_tool_surface(
            protocol_version=self._protocol_version,
            server_info=self._server_info,
            tools=tools,
        )

    def assert_tool_surface_unchanged(self) -> None:
        if self._tool_surface is None:
            raise RuntimeError("MCP client is not initialized")
        current = self._discover_tool_surface()
        if current != self._tool_surface:
            raise MCPToolSurfaceDriftError(
                "live MCP tools/list differs from the frozen cell surface"
            )

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if self._tool_surface is None:
            raise RuntimeError("MCP client is not initialized")
        names = {tool["name"] for tool in self._tool_surface["tools"]}
        if name not in names:
            raise MCPProtocolError(f"model requested unknown MCP tool: {name}")
        if not isinstance(arguments, dict):
            raise MCPProtocolError("MCP tool arguments must be an object")
        result = self._request(
            "tools/call",
            {"name": name, "arguments": arguments},
        )
        result_bytes = tool_result_text(result).encode()
        if len(result_bytes) > self.config.max_tool_result_bytes:
            raise MCPProtocolError("MCP tool result exceeds the model-visible byte limit")
        transcript_row = self.transcript[-1]
        if transcript_row.get("method") != "tools/call":
            raise MCPProtocolError("MCP protocol transcript lost tools/call ordering")
        transcript_row.update(
            {
                "tool_name": name,
                "arguments_sha256": _sha256(arguments),
                "is_error": result.get("isError") is True,
                "transport_profile": self.transport_profile,
            }
        )
        return result

    def close(self) -> None:
        if self._closed:
            return
        try:
            if self._session_id is not None:
                try:
                    status, _headers, body = self.transport.request(
                        method="DELETE",
                        url=self.config.endpoint,
                        headers=self._headers(),
                        body=None,
                        timeout_seconds=float(self.config.timeout_seconds),
                        max_response_bytes=self.config.max_response_bytes,
                    )
                    self.transcript.append(
                        {
                            "method": "session/delete",
                            "request_id": None,
                            "request_sha256": hashlib.sha256(b"").hexdigest(),
                            "http_status": int(status),
                            "response_sha256": hashlib.sha256(body).hexdigest(),
                        }
                    )
                except Exception as error:
                    self.transcript.append(
                        {
                            "method": "session/delete",
                            "request_id": None,
                            "request_sha256": hashlib.sha256(b"").hexdigest(),
                            "http_status": None,
                            "response_sha256": None,
                            "error_class": type(error).__name__,
                        }
                    )
        finally:
            self._closed = True

    def __enter__(self) -> "StreamableHTTPMCPClient":
        self.initialize()
        return self

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        self.close()


class StreamableHTTPMCPClientFactory:
    """Creates a fresh MCP protocol session for each benchmark cell."""

    def __init__(
        self,
        config: StreamableHTTPConfig,
        *,
        transport_factory: Callable[[], HTTPTransport] | None = None,
    ) -> None:
        self.config = config
        self._transport_factory = transport_factory
        probe_transport = (
            transport_factory() if transport_factory is not None else None
        )
        probe = StreamableHTTPMCPClient(config, transport=probe_transport)
        self.identity = {
            "factory_id": "aivideo-bench-streamable-http-mcp-factory",
            "factory_version": MCP_CLIENT_VERSION,
            "implementation_sha256": _implementation_sha256(),
            "configuration_sha256": probe.identity["configuration_sha256"],
            "credential_id": config.credential_id,
            "credential_material_sha256": probe.identity[
                "credential_material_sha256"
            ],
            "official_transport": (
                transport_factory is None
                and not config.allow_insecure_loopback
                and urlsplit(config.endpoint).scheme == "https"
            ),
        }

    def create(self) -> StreamableHTTPMCPClient:
        transport = (
            self._transport_factory()
            if self._transport_factory is not None
            else None
        )
        client = StreamableHTTPMCPClient(self.config, transport=transport)
        if client.identity["configuration_sha256"] != self.identity[
            "configuration_sha256"
        ]:
            raise RuntimeError("MCP client factory configuration identity drifted")
        return client


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def attest_mcp_runtime(
    factory: StreamableHTTPMCPClientFactory,
    expected_tool_surface: dict[str, Any],
    *,
    role: str,
) -> dict[str, Any]:
    """Prove endpoint auth and frozen-schema readiness without invoking a tool.

    This deliberately performs two complete tool-surface reads: initialization
    discovers the first surface, then ``assert_tool_surface_unchanged`` discovers
    it again. It makes no model, tool, rendering, or generation request.
    """

    if role not in MCP_RUNTIME_ROLES:
        raise ValueError("MCP runtime attestation role is invalid")
    surface_errors = validate_tool_surface(expected_tool_surface)
    if surface_errors:
        raise ValueError(
            "cannot attest an invalid MCP tool surface: " + "; ".join(surface_errors)
        )
    factory_identity = dict(getattr(factory, "identity", {}))
    expected_factory_keys = {
        "factory_id",
        "factory_version",
        "implementation_sha256",
        "configuration_sha256",
        "credential_id",
        "credential_material_sha256",
        "official_transport",
    }
    if set(factory_identity) != expected_factory_keys:
        raise ValueError("MCP runtime factory identity keys are malformed")
    if factory_identity.get("official_transport") is not True:
        raise ValueError("MCP runtime attestation requires the official HTTPS transport")
    for key in ("implementation_sha256", "configuration_sha256"):
        if not _is_sha256(factory_identity.get(key)):
            raise ValueError(f"MCP runtime factory {key} is malformed")
    credential_id = factory_identity.get("credential_id")
    if not isinstance(credential_id, str) or not credential_id.strip():
        raise ValueError("MCP runtime factory credential ID is missing")
    if not _is_sha256(factory_identity.get("credential_material_sha256")):
        raise ValueError(
            "MCP runtime factory actual credential commitment is missing"
        )

    client = factory.create()
    initialized = False
    surface_recheck_passed = False
    try:
        observed_surface = client.initialize()
        initialized = True
        if observed_surface != expected_tool_surface:
            raise MCPToolSurfaceDriftError(
                f"{role} live MCP surface differs from the frozen benchmark surface"
            )
        client.assert_tool_surface_unchanged()
        surface_recheck_passed = True
        if any(row.get("method") == "tools/call" for row in client.transcript):
            raise MCPProtocolError("MCP runtime preflight invoked a tool")
    finally:
        client.close()

    client_identity = dict(getattr(client, "identity", {}))
    if client_identity.get("configuration_sha256") != factory_identity[
        "configuration_sha256"
    ]:
        raise RuntimeError("MCP preflight client differs from its factory identity")
    attestation: dict[str, Any] = {
        "schema_version": MCP_RUNTIME_ATTESTATION_SCHEMA,
        "role": role,
        "factory_identity_sha256": _sha256(factory_identity),
        "factory_configuration_sha256": factory_identity["configuration_sha256"],
        "client_identity_sha256": _sha256(client_identity),
        "credential_id_sha256": _sha256({"credential_id": credential_id}),
        "credential_material_sha256": factory_identity[
            "credential_material_sha256"
        ],
        "transport_profile": client.transport_profile,
        "protocol_version": expected_tool_surface["protocol_version"],
        "server_info_sha256": _sha256(expected_tool_surface["server_info"]),
        "tool_surface_sha256": expected_tool_surface["tool_surface_sha256"],
        "initialized": initialized,
        "surface_recheck_passed": surface_recheck_passed,
        "llm_inference_calls": 0,
        "media_processing_calls": 0,
        "tool_calls": 0,
    }
    attestation["attestation_sha256"] = _sha256(attestation)
    return attestation


def validate_mcp_runtime_attestation(
    value: Any,
    *,
    role: str,
    expected_tool_surface: dict[str, Any],
    expected_credential_id: str,
) -> list[str]:
    """Validate the secret-free evidence emitted by ``attest_mcp_runtime``."""

    label = f"{role} MCP runtime attestation"
    if role not in MCP_RUNTIME_ROLES:
        return ["MCP runtime attestation role is invalid"]
    if not isinstance(value, dict) or set(value) != MCP_RUNTIME_ATTESTATION_KEYS:
        return [f"{label} keys do not match the canonical schema"]
    errors: list[str] = []
    if value.get("schema_version") != MCP_RUNTIME_ATTESTATION_SCHEMA:
        errors.append(f"{label} schema mismatch")
    if value.get("role") != role:
        errors.append(f"{label} role mismatch")
    for key in (
        "factory_identity_sha256",
        "factory_configuration_sha256",
        "client_identity_sha256",
        "credential_id_sha256",
        "credential_material_sha256",
        "server_info_sha256",
        "tool_surface_sha256",
        "attestation_sha256",
    ):
        if not _is_sha256(value.get(key)):
            errors.append(f"{label} {key} is malformed")
    expected_unhashed = {
        key: item for key, item in value.items() if key != "attestation_sha256"
    }
    if value.get("attestation_sha256") != _sha256(expected_unhashed):
        errors.append(f"{label} self-hash mismatch")
    if value.get("credential_id_sha256") != _sha256(
        {"credential_id": expected_credential_id}
    ):
        errors.append(f"{label} credential commitment mismatch")
    if value.get("transport_profile") not in MCP_TRANSPORT_PROFILES:
        errors.append(f"{label} transport profile is invalid")
    if value.get("protocol_version") != expected_tool_surface.get(
        "protocol_version"
    ):
        errors.append(f"{label} protocol version mismatch")
    if value.get("server_info_sha256") != _sha256(
        expected_tool_surface.get("server_info")
    ):
        errors.append(f"{label} server identity mismatch")
    if value.get("tool_surface_sha256") != expected_tool_surface.get(
        "tool_surface_sha256"
    ):
        errors.append(f"{label} tool-surface mismatch")
    if value.get("initialized") is not True:
        errors.append(f"{label} did not initialize")
    if value.get("surface_recheck_passed") is not True:
        errors.append(f"{label} did not pass the second surface read")
    for key in ("llm_inference_calls", "media_processing_calls", "tool_calls"):
        if value.get(key) != 0 or isinstance(value.get(key), bool):
            errors.append(f"{label} {key} must equal integer zero")
    return errors
