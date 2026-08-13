import copy
import contextlib
import hashlib
import http.server
import json
import threading
import unittest

from aivideo_bench.mcp_client import (
    MCPProtocolError,
    MCP_TRANSPORT_PROFILE_SERVER_SESSION,
    MCP_TRANSPORT_PROFILE_STATELESS,
    MCPToolSurfaceDriftError,
    StreamableHTTPConfig,
    StreamableHTTPMCPClient,
    StreamableHTTPMCPClientFactory,
    attest_mcp_runtime,
    build_tool_surface,
    openrouter_tools,
    tool_result_text,
    validate_streamable_http_config,
    validate_mcp_runtime_attestation,
    validate_tool_surface,
)
from aivideo_bench.standard import canonical_bytes


def _response(request_id, result):
    return json.dumps(
        {"jsonrpc": "2.0", "id": request_id, "result": result},
        separators=(",", ":"),
    ).encode()


def _tool(name, description="Read state"):
    return {
        "name": name,
        "description": description,
        "inputSchema": {
            "type": "object",
            "properties": {"project_id": {"type": "string"}},
            "required": ["project_id"],
            "additionalProperties": False,
        },
        "annotations": {"readOnlyHint": name == "get_context"},
    }


def _initialize_result():
    return {
        "protocolVersion": "2025-06-18",
        "capabilities": {"tools": {"listChanged": True}},
        "serverInfo": {
            "name": "aivideo-public-mcp",
            "title": "AIVideo",
            "version": "git-deadbeef",
        },
    }


class FakeHTTPTransport:
    max_attempts = 1

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def request(self, **kwargs):
        self.requests.append(copy.deepcopy(kwargs))
        if not self.responses:
            raise AssertionError("unexpected HTTP request")
        return self.responses.pop(0)


def _client(transport, **overrides):
    values = {
        "endpoint": "https://mcp.example.test/mcp",
        "credential_id": "benchmark-principal-egemen",
        "headers": (("Authorization", "Bearer TEST_ONLY_NOT_A_SECRET"),),
        "timeout_seconds": 30,
    }
    values.update(overrides)
    return StreamableHTTPMCPClient(
        StreamableHTTPConfig(**values),
        transport=transport,
    )


def _lifecycle(tool_pages, *, tool_result=None):
    responses = [
        (
            200,
            {
                "Content-Type": "application/json",
                "Mcp-Session-Id": "session-123",
            },
            _response(1, _initialize_result()),
        ),
        (202, {"Content-Type": "application/json"}, b""),
    ]
    request_id = 2
    for page in tool_pages:
        responses.append(
            (200, {"Content-Type": "application/json"}, _response(request_id, page))
        )
        request_id += 1
    if tool_result is not None:
        responses.append(
            (
                200,
                {"Content-Type": "application/json"},
                _response(request_id, tool_result),
            )
        )
    responses.append((200, {}, b""))
    return responses


@contextlib.contextmanager
def _local_mcp_server(tools=None):
    server_tools = copy.deepcopy(tools or [_tool("get_context")])

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, _format, *_args):
            return

        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            self.server.requests.append(
                {
                    "payload": payload,
                    "headers": {
                        key.casefold(): value for key, value in self.headers.items()
                    },
                }
            )
            method = payload["method"]
            if method == "notifications/initialized":
                self.send_response(202)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            if method == "initialize":
                result = _initialize_result()
            elif method == "tools/list":
                result = {"tools": copy.deepcopy(server_tools)}
            elif method == "tools/call":
                result = {
                    "content": [{"type": "text", "text": "local project"}],
                    "structuredContent": {"project_id": "local-p1"},
                    "isError": False,
                }
            else:
                self.send_error(400)
                return
            body = _response(payload["id"], result)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            if method == "initialize":
                self.send_header("Mcp-Session-Id", "local-session-1")
            self.end_headers()
            self.wfile.write(body)

        def do_DELETE(self):
            self.server.deletes += 1
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.requests = []
    server.deletes = 0
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


class MCPToolSurfaceTests(unittest.TestCase):
    def test_surface_is_order_independent_and_maps_exact_model_schema(self):
        first = build_tool_surface(
            protocol_version="2025-06-18",
            server_info={"name": "aivideo", "version": "1"},
            tools=[_tool("place_clips"), _tool("get_context")],
        )
        second = build_tool_surface(
            protocol_version="2025-06-18",
            server_info={"version": "1", "name": "aivideo"},
            tools=[_tool("get_context"), _tool("place_clips")],
        )
        self.assertEqual(first, second)
        self.assertEqual([], validate_tool_surface(first))
        model_tools = openrouter_tools(first)
        self.assertEqual(
            ["get_context", "place_clips"],
            [tool["function"]["name"] for tool in model_tools],
        )
        self.assertEqual(
            first["tools"][0]["inputSchema"],
            model_tools[0]["function"]["parameters"],
        )

    def test_surface_rejects_duplicates_unknown_fields_and_bad_commitment(self):
        with self.assertRaisesRegex(MCPProtocolError, "duplicate"):
            build_tool_surface(
                protocol_version="2025-06-18",
                server_info={"name": "aivideo", "version": "1"},
                tools=[_tool("get_context"), _tool("get_context")],
            )
        bad = _tool("get_context")
        bad["untrustedExtension"] = True
        with self.assertRaisesRegex(MCPProtocolError, "unsupported fields"):
            build_tool_surface(
                protocol_version="2025-06-18",
                server_info={"name": "aivideo", "version": "1"},
                tools=[bad],
            )
        valid = build_tool_surface(
            protocol_version="2025-06-18",
            server_info={"name": "aivideo", "version": "1"},
            tools=[_tool("get_context")],
        )
        valid["tool_surface_sha256"] = "0" * 64
        self.assertIn("commitment", validate_tool_surface(valid)[0])

    def test_current_protocol_tool_and_server_metadata_are_committed(self):
        tool = _tool("get_context")
        tool["icons"] = [
            {"src": "https://aivideo.example/icon.png", "mimeType": "image/png"}
        ]
        tool["execution"] = {"taskSupport": "forbidden"}
        surface = build_tool_surface(
            protocol_version="2025-11-25",
            server_info={
                "name": "aivideo",
                "version": "2",
                "description": "AIVideo MCP",
                "websiteUrl": "https://aivideo.com",
            },
            tools=[tool],
        )
        self.assertEqual([], validate_tool_surface(surface))
        self.assertEqual("forbidden", surface["tools"][0]["execution"]["taskSupport"])
        self.assertEqual("AIVideo MCP", surface["server_info"]["description"])

    def test_tool_result_is_lossless_and_strict(self):
        result = {
            "content": [{"type": "text", "text": "ok"}],
            "structuredContent": {"asset_id": "asset_1"},
            "isError": False,
        }
        self.assertEqual(result, json.loads(tool_result_text(result)))
        with self.assertRaisesRegex(MCPProtocolError, "content"):
            tool_result_text({"isError": False})


class StreamableHTTPMCPClientTests(unittest.TestCase):
    @staticmethod
    def _attestation_factory(responses):
        class Factory:
            def __init__(self):
                probe = _client(FakeHTTPTransport([]))
                self.identity = {
                    "factory_id": "aivideo-bench-streamable-http-mcp-factory",
                    "factory_version": "1",
                    "implementation_sha256": "a" * 64,
                    "configuration_sha256": probe.identity[
                        "configuration_sha256"
                    ],
                    "credential_id": "benchmark-principal-egemen",
                    "credential_material_sha256": probe.identity[
                        "credential_material_sha256"
                    ],
                    "official_transport": True,
                }
                self.clients = []

            def create(self):
                client = _client(FakeHTTPTransport(copy.deepcopy(responses)))
                self.clients.append(client)
                return client

        return Factory()

    def test_runtime_attestation_double_reads_surface_without_tool_or_paid_calls(self):
        tools = [_tool("get_context"), _tool("place_clips")]
        expected = build_tool_surface(
            protocol_version="2025-06-18",
            server_info=_initialize_result()["serverInfo"],
            tools=tools,
        )
        responses = _lifecycle([{"tools": tools}])[:-1]
        responses.extend(
            [
                (
                    200,
                    {"Content-Type": "application/json"},
                    _response(3, {"tools": tools}),
                ),
                (200, {}, b""),
            ]
        )
        factory = self._attestation_factory(responses)
        attestation = attest_mcp_runtime(
            factory,
            expected,
            role="candidate",
        )
        self.assertEqual(
            [],
            validate_mcp_runtime_attestation(
                attestation,
                role="candidate",
                expected_tool_surface=expected,
                expected_credential_id="benchmark-principal-egemen",
            ),
        )
        methods = [row["method"] for row in factory.clients[0].transcript]
        self.assertEqual(
            [
                "initialize",
                "notifications/initialized",
                "tools/list",
                "tools/list",
                "session/delete",
            ],
            methods,
        )
        self.assertEqual(0, attestation["llm_inference_calls"])
        self.assertEqual(0, attestation["media_processing_calls"])
        self.assertEqual(0, attestation["tool_calls"])
        self.assertEqual(
            MCP_TRANSPORT_PROFILE_SERVER_SESSION,
            attestation["transport_profile"],
        )

        tampered = copy.deepcopy(attestation)
        tampered["transport_profile"] = "invented_transport"
        unhashed = {
            key: value
            for key, value in tampered.items()
            if key != "attestation_sha256"
        }
        tampered["attestation_sha256"] = hashlib.sha256(
            canonical_bytes(unhashed)
        ).hexdigest()
        self.assertTrue(
            any(
                "transport profile is invalid" in error
                for error in validate_mcp_runtime_attestation(
                    tampered,
                    role="candidate",
                    expected_tool_surface=expected,
                    expected_credential_id="benchmark-principal-egemen",
                )
            )
        )

    def test_transport_profile_is_observed_and_frozen_after_initialization(self):
        tools = [_tool("get_context")]
        stateless_responses = [
            (
                200,
                {"Content-Type": "application/json"},
                _response(1, _initialize_result()),
            ),
            (202, {"Content-Type": "application/json"}, b""),
            (
                200,
                {"Content-Type": "application/json"},
                _response(2, {"tools": tools}),
            ),
        ]
        client = _client(FakeHTTPTransport(stateless_responses))
        client.initialize()
        self.assertEqual(MCP_TRANSPORT_PROFILE_STATELESS, client.transport_profile)
        client.close()
        self.assertNotIn(
            "session/delete", [row["method"] for row in client.transcript]
        )

        tool_result = {
            "content": [{"type": "text", "text": "ok"}],
            "structuredContent": {"project_id": "project-1"},
            "isError": False,
        }
        late_session_responses = [
            *stateless_responses,
            (
                200,
                {
                    "Content-Type": "application/json",
                    "Mcp-Session-Id": "late-session",
                },
                _response(3, tool_result),
            ),
        ]
        client = _client(FakeHTTPTransport(late_session_responses))
        client.initialize()
        with self.assertRaisesRegex(MCPProtocolError, "introduced a session"):
            client.call_tool("get_context", {"project_id": "project-1"})

    def test_runtime_attestation_fails_closed_on_second_surface_read_drift(self):
        tools = [_tool("get_context")]
        expected = build_tool_surface(
            protocol_version="2025-06-18",
            server_info=_initialize_result()["serverInfo"],
            tools=tools,
        )
        responses = _lifecycle([{"tools": tools}])[:-1]
        responses.extend(
            [
                (
                    200,
                    {"Content-Type": "application/json"},
                    _response(
                        3,
                        {"tools": [*tools, _tool("place_clips")]},
                    ),
                ),
                (200, {}, b""),
            ]
        )
        factory = self._attestation_factory(responses)
        with self.assertRaises(MCPToolSurfaceDriftError):
            attest_mcp_runtime(factory, expected, role="observer")
        self.assertEqual("session/delete", factory.clients[0].transcript[-1]["method"])

    def test_loopback_streamable_http_integration_without_paid_calls(self):
        with _local_mcp_server() as server:
            endpoint = f"http://127.0.0.1:{server.server_port}/mcp"
            config = StreamableHTTPConfig(
                endpoint=endpoint,
                credential_id="local-integration-principal",
                headers=(("Authorization", "Bearer LOCAL_TEST_ONLY"),),
                allow_insecure_loopback=True,
            )
            with StreamableHTTPMCPClient(config) as client:
                result = client.call_tool(
                    "get_context", {"project_id": "local-p1"}
                )
                self.assertEqual(
                    "local-p1", result["structuredContent"]["project_id"]
                )
            self.assertEqual(1, server.deletes)
            methods = [request["payload"]["method"] for request in server.requests]
            self.assertEqual(
                [
                    "initialize",
                    "notifications/initialized",
                    "tools/list",
                    "tools/call",
                ],
                methods,
            )
            self.assertEqual(
                "local-session-1",
                server.requests[-1]["headers"]["mcp-session-id"],
            )
            self.assertEqual(
                "2025-06-18",
                server.requests[-1]["headers"]["mcp-protocol-version"],
            )

    def test_initializes_paginates_calls_real_mcp_and_closes_session(self):
        pages = [
            {"tools": [_tool("place_clips")], "nextCursor": "page-2"},
            {"tools": [_tool("get_context")]},
        ]
        result = {
            "content": [{"type": "text", "text": '{"project":{"id":"p1"}}'}],
            "isError": False,
        }
        transport = FakeHTTPTransport(_lifecycle(pages, tool_result=result))
        client = _client(transport)
        surface = client.initialize()
        self.assertEqual(
            ["get_context", "place_clips"],
            [tool["name"] for tool in surface["tools"]],
        )
        self.assertEqual(result, client.call_tool("get_context", {"project_id": "p1"}))
        client.close()

        methods = []
        for request in transport.requests:
            if request["body"]:
                methods.append(json.loads(request["body"])["method"])
            else:
                methods.append("DELETE")
        self.assertEqual(
            [
                "initialize",
                "notifications/initialized",
                "tools/list",
                "tools/list",
                "tools/call",
                "DELETE",
            ],
            methods,
        )
        self.assertNotIn("MCP-Protocol-Version", transport.requests[0]["headers"])
        self.assertEqual(
            "2025-11-25",
            json.loads(transport.requests[0]["body"])["params"]["protocolVersion"],
        )
        for request in transport.requests[1:]:
            self.assertEqual(
                "2025-06-18", request["headers"]["MCP-Protocol-Version"]
            )
            self.assertEqual("session-123", request["headers"]["Mcp-Session-Id"])
        self.assertEqual(
            "page-2",
            json.loads(transport.requests[3]["body"])["params"]["cursor"],
        )

    def test_secret_is_excluded_from_identity_and_evidence(self):
        transport = FakeHTTPTransport(
            _lifecycle([{"tools": [_tool("get_context")]}])
        )
        client = _client(transport)
        serialized_identity = json.dumps(client.identity, sort_keys=True)
        self.assertNotIn("TEST_ONLY_NOT_A_SECRET", serialized_identity)
        client.initialize()
        self.assertNotIn(
            "TEST_ONLY_NOT_A_SECRET", json.dumps(client.transcript, sort_keys=True)
        )
        client.close()

    def test_unknown_tool_is_rejected_without_http_request(self):
        transport = FakeHTTPTransport(
            _lifecycle([{"tools": [_tool("get_context")]}])
        )
        client = _client(transport)
        client.initialize()
        count = len(transport.requests)
        with self.assertRaisesRegex(MCPProtocolError, "unknown MCP tool"):
            client.call_tool("delete_everything", {})
        self.assertEqual(count, len(transport.requests))
        client.close()

    def test_session_close_failure_is_evidence_not_cell_failure(self):
        responses = _lifecycle([{"tools": [_tool("get_context")]}])[:-1]
        transport = FakeHTTPTransport(responses)
        client = _client(transport)
        client.initialize()
        client.close()
        self.assertTrue(client._closed)
        self.assertEqual(
            "AssertionError", client.transcript[-1]["error_class"]
        )

    def test_tool_surface_drift_is_detected(self):
        responses = _lifecycle([{"tools": [_tool("get_context")]}])[:-1]
        responses.append(
            (
                200,
                {"Content-Type": "application/json"},
                _response(3, {"tools": [_tool("get_context"), _tool("place_clips")]}),
            )
        )
        responses.append((200, {}, b""))
        transport = FakeHTTPTransport(responses)
        client = _client(transport)
        client.initialize()
        with self.assertRaises(MCPToolSurfaceDriftError):
            client.assert_tool_surface_unchanged()
        client.close()

    def test_sse_response_and_tool_error_are_model_visible(self):
        init = _response(1, _initialize_result()).decode()
        list_result = _response(2, {"tools": [_tool("get_context")]}).decode()
        error_result = _response(
            3,
            {
                "content": [{"type": "text", "text": "project not found"}],
                "isError": True,
            },
        ).decode()
        transport = FakeHTTPTransport(
            [
                (
                    200,
                    {"Content-Type": "text/event-stream", "Mcp-Session-Id": "sse-1"},
                    f"event: message\ndata: {init}\n\n".encode(),
                ),
                (202, {}, b""),
                (
                    200,
                    {"Content-Type": "text/event-stream"},
                    f": heartbeat\n\nevent: message\ndata: {list_result}\n\n".encode(),
                ),
                (
                    200,
                    {"Content-Type": "text/event-stream"},
                    f"event: message\ndata: {error_result}\n\n".encode(),
                ),
                (200, {}, b""),
            ]
        )
        client = _client(transport)
        client.initialize()
        result = client.call_tool("get_context", {"project_id": "missing"})
        self.assertIs(result["isError"], True)
        client.close()

    def test_oversized_tool_result_is_rejected_before_model_exposure(self):
        result = {
            "content": [{"type": "text", "text": "x" * 1500}],
            "isError": False,
        }
        transport = FakeHTTPTransport(
            _lifecycle([{"tools": [_tool("get_context")]}], tool_result=result)
        )
        client = _client(
            transport,
            max_response_bytes=4096,
            max_total_response_bytes=16384,
            max_tool_result_bytes=1024,
        )
        client.initialize()
        with self.assertRaisesRegex(MCPProtocolError, "model-visible byte limit"):
            client.call_tool("get_context", {"project_id": "p1"})
        client.close()

    def test_config_rejects_insecure_or_secret_bearing_endpoint_and_retries(self):
        for endpoint in (
            "http://mcp.example.test/mcp",
            "https://user:pass@mcp.example.test/mcp",
            "https://mcp.example.test/mcp?token=secret",
        ):
            errors = validate_streamable_http_config(
                StreamableHTTPConfig(endpoint=endpoint, credential_id="principal-1")
            )
            self.assertTrue(errors, endpoint)
        localhost = StreamableHTTPConfig(
            endpoint="http://127.0.0.1:8000/mcp",
            credential_id="principal-1",
            allow_insecure_loopback=True,
        )
        self.assertEqual([], validate_streamable_http_config(localhost))
        self.assertFalse(
            StreamableHTTPMCPClientFactory(localhost).identity[
                "official_transport"
            ]
        )
        bad_limits = StreamableHTTPConfig(
            endpoint="https://mcp.example.test/mcp",
            credential_id="principal-1",
            max_response_bytes=4096,
            max_total_response_bytes=2048,
        )
        self.assertTrue(validate_streamable_http_config(bad_limits))

        class RetryingTransport(FakeHTTPTransport):
            max_attempts = 2

        with self.assertRaisesRegex(ValueError, "exactly one"):
            _client(RetryingTransport([]))

    def test_factory_creates_fresh_secret_safe_cell_clients(self):
        transports = []

        def make_transport():
            transport = FakeHTTPTransport([])
            transports.append(transport)
            return transport

        config = StreamableHTTPConfig(
            endpoint="https://mcp.example.test/mcp",
            credential_id="benchmark-principal-egemen",
            headers=(("Authorization", "Bearer TEST_ONLY_NOT_A_SECRET"),),
        )
        factory = StreamableHTTPMCPClientFactory(
            config,
            transport_factory=make_transport,
        )
        first = factory.create()
        second = factory.create()
        self.assertIsNot(first, second)
        self.assertFalse(factory.identity["official_transport"])
        self.assertNotIn(
            "TEST_ONLY_NOT_A_SECRET", json.dumps(factory.identity, sort_keys=True)
        )
        self.assertEqual(3, len(transports))


if __name__ == "__main__":
    unittest.main()
