"""Bounded OpenRouter candidate loop over the real AIVideo MCP surface."""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import math
import time
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

from .json_schema_validation import (
    validate_json_schema_instance,
    validate_schema_vocabulary,
)
from .mcp_client import (
    MCPToolSurfaceDriftError,
    StreamableHTTPMCPClientFactory,
    tool_result_text,
    validate_tool_surface,
)
from .standard import canonical_bytes, sha256


OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
SYSTEM_PROMPT = """You are being evaluated as a professional media agent.
Complete the private task by inspecting and editing only the assigned AIVideo
project with the provided AIVideo MCP tools. A prose plan is not completion.
Treat IDs and state as mutable; inspect context instead of guessing. Respect the
tool allowlist, verify every tool result, preserve unrelated work, and finish
with one short status sentence. Never claim an action that a tool did not
confirm."""
PROMPT_POLICY_SHA256 = sha256(
    {"version": "aivideo-bench-candidate-prompt-v1", "system_prompt": SYSTEM_PROMPT}
)


class CompletionTransport(Protocol):
    max_attempts: int
    identity: Mapping[str, Any]

    def serialize(self, payload: Mapping[str, Any]) -> bytes: ...

    def complete(self, payload: Mapping[str, Any]) -> dict[str, Any]: ...


class MCPClientLike(Protocol):
    transcript: list[dict[str, Any]]

    def initialize(self) -> dict[str, Any]: ...

    def assert_tool_surface_unchanged(self) -> None: ...

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]: ...

    def close(self) -> None: ...


class MCPFactoryLike(Protocol):
    identity: Mapping[str, Any]

    def create(self) -> MCPClientLike: ...


class SingleAttemptOpenRouterTransport:
    """OpenRouter transport with no implicit retry of a paid request."""

    max_attempts = 1

    def __init__(self, api_key: str, *, timeout_seconds: float = 300.0) -> None:
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("OpenRouter API key must be non-empty")
        if not 0 < float(timeout_seconds) <= 1800:
            raise ValueError("OpenRouter timeout must be in (0,1800]")
        self._api_key = api_key
        self.timeout_seconds = float(timeout_seconds)
        self.identity = {
            "transport": "openrouter-chat-completions",
            "version": "1",
            "api_url_sha256": hashlib.sha256(OPENROUTER_API_URL.encode()).hexdigest(),
            "timeout_seconds": self.timeout_seconds,
        }

    @staticmethod
    def serialize(payload: Mapping[str, Any]) -> bytes:
        return canonical_bytes(payload)

    def complete(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            OPENROUTER_API_URL,
            data=self.serialize(payload),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://aivideo.com",
                "X-Title": "AIVideo Bench",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.timeout_seconds
            ) as response:
                value = json.load(response)
        except urllib.error.HTTPError as error:
            body = error.read(1000).decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenRouter HTTP {error.code}: {body}") from error
        if not isinstance(value, dict):
            raise RuntimeError("OpenRouter response must be an object")
        return value


@dataclass(frozen=True)
class CandidateConfig:
    model_id: str
    exact_model_id: str
    provider_slug: str
    reasoning_effort: str = "medium"
    max_reasoning_tokens: int = 4096
    max_completion_tokens: int = 4096
    max_steps: int = 12
    max_tool_calls: int = 24
    max_cost_per_task_usd: float = 0.25
    max_prompt_price_per_million: float = 5.0
    max_completion_price_per_million: float = 20.0
    max_request_bytes: int = 2 * 1024 * 1024
    max_tool_media_bytes: int = 4 * 1024 * 1024


def validate_candidate_config(config: CandidateConfig) -> list[str]:
    errors: list[str] = []
    for key in ("model_id", "exact_model_id", "provider_slug", "reasoning_effort"):
        value = getattr(config, key)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"candidate {key} must be non-empty")
    for key, lower, upper in (
        ("max_reasoning_tokens", 0, 131072),
        ("max_completion_tokens", 1, 131072),
        ("max_steps", 1, 64),
        ("max_tool_calls", 1, 256),
        ("max_request_bytes", 1024, 16 * 1024 * 1024),
        ("max_tool_media_bytes", 0, 64 * 1024 * 1024),
    ):
        value = getattr(config, key)
        if isinstance(value, bool) or not isinstance(value, int) or not lower <= value <= upper:
            errors.append(f"candidate {key} must be an integer in [{lower},{upper}]")
    for key in (
        "max_cost_per_task_usd",
        "max_prompt_price_per_million",
        "max_completion_price_per_million",
    ):
        value = getattr(config, key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0
        ):
            errors.append(f"candidate {key} must be finite and nonnegative")
    return errors


def _tool_map(surface: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    errors = validate_tool_surface(surface)
    if errors:
        raise ValueError("invalid MCP tool surface: " + "; ".join(errors))
    return {tool["name"]: tool for tool in surface["tools"]}


def _openrouter_tool(tool: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool["description"],
            "parameters": tool["inputSchema"],
        },
    }


def _project_bindings(value: Any, *, path: str = "$") -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            child_path = f"{path}.{key}"
            if key.replace("_", "").replace("-", "").casefold() == "projectid":
                if not isinstance(child, str) or not child.strip():
                    raise ValueError(f"{child_path} must be a non-empty project ID")
                rows.append((child_path, child))
            rows.extend(_project_bindings(child, path=child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            rows.extend(_project_bindings(child, path=f"{path}[{index}]"))
    return rows


def _project_scope_errors(
    arguments: Mapping[str, Any],
    *,
    assigned_project_id: str,
    require_binding: bool,
) -> list[str]:
    try:
        bindings = _project_bindings(arguments)
    except ValueError as error:
        return [str(error)]
    errors: list[str] = []
    if require_binding and not bindings:
        errors.append("mutating MCP call has no project ID binding")
    foreign = [path for path, value in bindings if value != assigned_project_id]
    if foreign:
        errors.append("MCP call references a foreign project at " + ", ".join(foreign))
    return errors


def _assistant_message(message: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "role": "assistant",
        "content": str(message.get("content") or ""),
    }
    if message.get("tool_calls"):
        result["tool_calls"] = copy.deepcopy(message["tool_calls"])
    return result


def _tool_media_blocks(result: Mapping[str, Any], *, byte_limit: int) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    total = 0
    for part in result.get("content") or []:
        if not isinstance(part, Mapping):
            continue
        kind = part.get("type")
        data = part.get("data")
        if kind not in {"image", "audio"} or not isinstance(data, str):
            continue
        try:
            decoded = base64.b64decode(data, validate=True)
        except ValueError as error:
            raise ValueError("MCP media block is not valid base64") from error
        total += len(decoded)
        if total > byte_limit:
            raise ValueError("MCP media blocks exceed the task byte limit")
        mime_type = str(part.get("mimeType") or "")
        if kind == "image" and mime_type.startswith("image/"):
            blocks.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{data}"},
                }
            )
        elif kind == "audio" and mime_type in {"audio/mpeg", "audio/wav"}:
            blocks.append(
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": data,
                        "format": "mp3" if mime_type == "audio/mpeg" else "wav",
                    },
                }
            )
    return blocks


def _response_message(
    response: Mapping[str, Any], *, config: CandidateConfig
) -> tuple[dict[str, Any], float, dict[str, Any]]:
    if response.get("error"):
        raise RuntimeError("OpenRouter returned a provider error")
    if response.get("model") != config.exact_model_id:
        raise RuntimeError("OpenRouter resolved a different model than the frozen ID")
    if response.get("provider") != config.provider_slug:
        raise RuntimeError("OpenRouter resolved a different provider than the frozen slug")
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise RuntimeError("OpenRouter response must contain exactly one choice")
    choice = choices[0]
    message = choice.get("message") if isinstance(choice, Mapping) else None
    if not isinstance(message, dict):
        raise RuntimeError("OpenRouter choice has no assistant message")
    usage = response.get("usage")
    cost = usage.get("cost") if isinstance(usage, Mapping) else None
    if (
        isinstance(cost, bool)
        or not isinstance(cost, (int, float))
        or not math.isfinite(float(cost))
        or float(cost) < 0
    ):
        raise RuntimeError("OpenRouter response lacks nonnegative cost data")
    evidence = {
        "response_id": response.get("id"),
        "response_sha256": hashlib.sha256(canonical_bytes(response)).hexdigest(),
        "resolved_model": response.get("model"),
        "provider": response.get("provider"),
        "finish_reason": choice.get("finish_reason"),
        "usage": copy.deepcopy(dict(usage)),
    }
    return message, float(cost), evidence


def _error_tool_message(tool_call_id: str, name: str, error: str) -> dict[str, Any]:
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "name": name,
        "content": tool_result_text(
            {
                "content": [{"type": "text", "text": error}],
                "isError": True,
            }
        ),
    }


def _usage_totals(api_calls: list[Mapping[str, Any]]) -> dict[str, int]:
    """Normalize common OpenRouter token fields without trusting one provider shape."""

    totals = defaultdict(int)
    for call in api_calls:
        usage = call.get("usage")
        if not isinstance(usage, Mapping):
            continue
        prompt_details = usage.get("prompt_tokens_details")
        completion_details = usage.get("completion_tokens_details")
        totals["input_tokens"] += int(
            usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0
        )
        totals["output_tokens"] += int(
            usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0
        )
        totals["cached_input_tokens"] += int(
            (
                prompt_details.get("cached_tokens", 0)
                if isinstance(prompt_details, Mapping)
                else usage.get("cache_read_input_tokens", 0)
            )
            or 0
        )
        totals["reasoning_tokens"] += int(
            (
                completion_details.get("reasoning_tokens", 0)
                if isinstance(completion_details, Mapping)
                else usage.get("reasoning_tokens", 0)
            )
            or 0
        )
    return {
        key: totals[key]
        for key in (
            "input_tokens",
            "cached_input_tokens",
            "reasoning_tokens",
            "output_tokens",
        )
    }


def run_candidate_task(
    private_task: Mapping[str, Any],
    *,
    pack: Mapping[str, Any],
    assigned_project_id: str,
    config: CandidateConfig,
    completion: CompletionTransport,
    mcp_factory: MCPFactoryLike,
) -> dict[str, Any]:
    """Run one bounded candidate task and return a secret-free execution receipt."""

    config_errors = validate_candidate_config(config)
    if config_errors:
        raise ValueError("invalid candidate config: " + "; ".join(config_errors))
    if completion.max_attempts != 1:
        raise ValueError("candidate completion transport must use exactly one attempt")
    if not assigned_project_id.strip():
        raise ValueError("assigned project ID must be non-empty")
    surface_policy = pack.get("tool_surface")
    if not isinstance(surface_policy, Mapping):
        raise ValueError("private pack has no tool-surface policy")
    allowed_names = private_task.get("allowed_tools")
    if not isinstance(allowed_names, list) or not allowed_names:
        raise ValueError("private task has no tool allowlist")
    zero_cost_names = set(surface_policy.get("zero_media_cost_tools") or [])
    if not set(allowed_names) <= zero_cost_names:
        raise ValueError("task allowlist contains a non-zero-media-cost tool")

    client = mcp_factory.create()
    status = "harness_error"
    error: str | None = None
    final_text = ""
    inference_calls = 0
    tool_call_attempts = 0
    spent = 0.0
    api_calls: list[dict[str, Any]] = []
    request_hashes: list[str] = []
    inference_seconds = 0.0
    mcp_seconds = 0.0
    first_model_response_seconds: float | None = None
    tool_metrics: dict[str, defaultdict[str, float]] = {}
    outstanding_tool_errors: defaultdict[str, int] = defaultdict(int)
    started = time.monotonic()
    try:
        surface = client.initialize()
        if surface.get("tool_surface_sha256") != surface_policy.get("sha256"):
            raise MCPToolSurfaceDriftError(
                "live MCP tool surface differs from the sealed private pack"
            )
        tools = _tool_map(surface)
        if sorted(tools) != surface_policy.get("tool_names"):
            raise MCPToolSurfaceDriftError("live MCP tool names differ from the sealed pack")
        missing = sorted(set(allowed_names) - set(tools))
        if missing:
            raise MCPToolSurfaceDriftError(f"task tools are absent from live MCP: {missing}")
        for name in allowed_names:
            schema_errors = validate_schema_vocabulary(tools[name]["inputSchema"])
            if schema_errors:
                raise ValueError(
                    f"MCP tool {name} uses unsupported schema vocabulary: "
                    + "; ".join(schema_errors)
                )
        model_tools = [_openrouter_tool(tools[name]) for name in allowed_names]
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": private_task["prompt"]},
        ]
        for step in range(1, config.max_steps + 1):
            payload = {
                "model": config.model_id,
                "messages": messages,
                "tools": model_tools,
                "tool_choice": "auto",
                "reasoning": {
                    "effort": config.reasoning_effort,
                    "max_tokens": config.max_reasoning_tokens,
                    "exclude": True,
                },
                "max_tokens": config.max_completion_tokens,
                "provider": {
                    "only": [config.provider_slug],
                    "order": [config.provider_slug],
                    "allow_fallbacks": False,
                    "require_parameters": True,
                    "data_collection": "deny",
                },
                "usage": {"include": True},
            }
            request_bytes = completion.serialize(payload)
            if len(request_bytes) > config.max_request_bytes:
                status = "request_too_large"
                error = "serialized model request exceeds the byte limit"
                break
            maximum_call_cost = (
                len(request_bytes)
                * config.max_prompt_price_per_million
                / 1_000_000
                + config.max_completion_tokens
                * config.max_completion_price_per_million
                / 1_000_000
            )
            if spent + maximum_call_cost > config.max_cost_per_task_usd + 1e-12:
                status = "cost_preflight_blocked"
                error = "conservative request bound exceeds the task cost ceiling"
                break
            request_hashes.append(hashlib.sha256(request_bytes).hexdigest())
            inference_started = time.monotonic()
            response = completion.complete(payload)
            inference_seconds += time.monotonic() - inference_started
            inference_calls += 1
            if first_model_response_seconds is None:
                first_model_response_seconds = time.monotonic() - started
            message, call_cost, call_evidence = _response_message(response, config=config)
            spent += call_cost
            api_calls.append({"step": step, **call_evidence})
            if spent > config.max_cost_per_task_usd + 1e-12:
                status = "cost_cap_exceeded"
                error = "reported inference cost exceeds the task cost ceiling"
                break
            messages.append(_assistant_message(message))
            tool_calls = message.get("tool_calls") or []
            if not isinstance(tool_calls, list):
                raise ValueError("assistant tool_calls must be a list")
            if not tool_calls:
                final_text = str(message.get("content") or "")
                status = "completed"
                break
            if tool_call_attempts + len(tool_calls) > config.max_tool_calls:
                status = "tool_limit_exceeded"
                error = "candidate exceeded the task tool-call limit"
                break
            pending_media: list[dict[str, Any]] = []
            for raw_call in tool_calls:
                if not isinstance(raw_call, Mapping):
                    raise ValueError("assistant tool call must be an object")
                function = raw_call.get("function")
                if not isinstance(function, Mapping):
                    raise ValueError("assistant tool call has no function object")
                name = str(function.get("name") or "<missing>")
                call_id = str(raw_call.get("id") or f"step-{step}-call-{tool_call_attempts + 1}")
                tool_call_attempts += 1
                aggregate = tool_metrics.setdefault(name, defaultdict(float))
                aggregate["attempts"] += 1
                if name not in allowed_names:
                    status = "tool_policy_violation"
                    error = f"candidate requested tool outside allowlist: {name}"
                    break
                try:
                    arguments = json.loads(function.get("arguments") or "{}")
                except json.JSONDecodeError as parse_error:
                    aggregate["invalid_arguments"] += 1
                    outstanding_tool_errors[name] += 1
                    messages.append(
                        _error_tool_message(
                            call_id, name, f"invalid_tool_arguments:{parse_error}"
                        )
                    )
                    continue
                if not isinstance(arguments, dict):
                    aggregate["invalid_arguments"] += 1
                    outstanding_tool_errors[name] += 1
                    messages.append(
                        _error_tool_message(
                            call_id, name, "invalid_tool_arguments:expected object"
                        )
                    )
                    continue
                schema_errors = validate_json_schema_instance(
                    arguments, tools[name]["inputSchema"]
                )
                if schema_errors:
                    aggregate["invalid_arguments"] += 1
                    outstanding_tool_errors[name] += 1
                    messages.append(
                        _error_tool_message(
                            call_id,
                            name,
                            "invalid_tool_arguments:" + "; ".join(schema_errors),
                        )
                    )
                    continue
                annotations = tools[name].get("annotations") or {}
                scope_errors = _project_scope_errors(
                    arguments,
                    assigned_project_id=assigned_project_id,
                    require_binding=annotations.get("readOnlyHint") is not True,
                )
                if scope_errors:
                    status = "project_scope_violation"
                    error = "; ".join(scope_errors)
                    break
                aggregate["dispatched"] += 1
                tool_started = time.monotonic()
                raw_result = client.call_tool(name, arguments)
                tool_latency = time.monotonic() - tool_started
                mcp_seconds += tool_latency
                aggregate["latency_seconds"] += tool_latency
                if raw_result.get("isError") is True:
                    aggregate["errors"] += 1
                    outstanding_tool_errors[name] += 1
                else:
                    aggregate["successes"] += 1
                    if outstanding_tool_errors[name]:
                        aggregate["same_tool_successes_after_error"] += 1
                        outstanding_tool_errors[name] -= 1
                pending_media.extend(
                    _tool_media_blocks(
                        raw_result, byte_limit=config.max_tool_media_bytes
                    )
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": name,
                        "content": tool_result_text(raw_result),
                    }
                )
            if status in {"tool_policy_violation", "project_scope_violation"}:
                break
            if pending_media:
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Authoritative media returned by the preceding MCP tools:",
                            },
                            *pending_media,
                        ],
                    }
                )
        else:
            status = "step_limit_exceeded"
            error = "candidate did not finish within the task step limit"
        client.assert_tool_surface_unchanged()
    finally:
        client.close()
    transcript = copy.deepcopy(client.transcript)
    return {
        "schema_version": "aivideo-bench-candidate-execution-v2",
        "task_id": private_task["task_id"],
        "blueprint_sha256": private_task["blueprint_sha256"],
        "pack_sha256": pack["pack_sha256"],
        "tool_surface_sha256": surface_policy["sha256"],
        "candidate": asdict(config),
        "prompt_policy_sha256": PROMPT_POLICY_SHA256,
        "transport_identity": copy.deepcopy(dict(completion.identity)),
        "mcp_factory_identity": copy.deepcopy(dict(mcp_factory.identity)),
        "status": status,
        "error": error,
        "final_text": final_text,
        "inference_calls": inference_calls,
        "tool_call_attempts": tool_call_attempts,
        "mcp_tool_calls": sum(row.get("method") == "tools/call" for row in transcript),
        "llm_inference_cost_usd": round(spent, 9),
        "paid_media_credits": 0.0,
        # The official cost gate covers both feasibility before a call and the
        # measured spend afterward. A preflight-blocked task is economically
        # invalid even though it correctly spends zero dollars.
        "cost_cap_respected": status
        not in {"cost_preflight_blocked", "cost_cap_exceeded"}
        and spent <= config.max_cost_per_task_usd + 1e-12,
        "request_sha256": request_hashes,
        "api_calls": api_calls,
        "mcp_transcript": transcript,
        "token_usage": _usage_totals(api_calls),
        "tool_metrics": [
            {
                "name": name,
                **{
                    key: int(aggregate[key])
                    for key in (
                        "attempts",
                        "dispatched",
                        "successes",
                        "errors",
                        "invalid_arguments",
                        "same_tool_successes_after_error",
                    )
                },
                "latency_seconds": round(aggregate["latency_seconds"], 6),
            }
            for name, aggregate in sorted(tool_metrics.items())
        ],
        "first_model_response_seconds": (
            None
            if first_model_response_seconds is None
            else round(first_model_response_seconds, 6)
        ),
        "inference_seconds": round(inference_seconds, 6),
        "mcp_seconds": round(mcp_seconds, 6),
        "latency_seconds": round(time.monotonic() - started, 6),
    }
