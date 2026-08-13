import base64
import copy
import json

import pytest

from aivideo_bench.candidate_execution import (
    CandidateConfig,
    run_candidate_task,
)
from aivideo_bench.mcp_client import build_tool_surface
from aivideo_bench.standard import canonical_bytes, standard


def _tool(name, *, read_only=False):
    return {
        "name": name,
        "description": f"Use {name}",
        "inputSchema": {
            "type": "object",
            "properties": {
                "project_id": {"type": "string", "minLength": 1},
                "clips": {"type": "array", "items": {"type": "object"}},
            },
            "required": ["project_id"],
            "additionalProperties": False,
        },
        "annotations": {"readOnlyHint": read_only},
    }


SURFACE = build_tool_surface(
    protocol_version="2025-06-18",
    server_info={"name": "aivideo", "version": "test"},
    tools=[_tool("get_context", read_only=True), _tool("place_clips")],
)


class FakeMCPClient:
    def __init__(self, surface=SURFACE, *, media=False):
        self.surface = copy.deepcopy(surface)
        self.media = media
        self.calls = []
        self.transcript = []
        self.closed = False
        self.rechecks = 0

    def initialize(self):
        self.transcript.extend(
            [
                {"method": "initialize"},
                {"method": "notifications/initialized"},
                {"method": "tools/list"},
            ]
        )
        return copy.deepcopy(self.surface)

    def assert_tool_surface_unchanged(self):
        self.rechecks += 1

    def call_tool(self, name, arguments):
        self.calls.append((name, copy.deepcopy(arguments)))
        self.transcript.append({"method": "tools/call", "tool_name": name})
        content = [{"type": "text", "text": "ok"}]
        if self.media:
            content.append(
                {
                    "type": "image",
                    "mimeType": "image/png",
                    "data": base64.b64encode(b"small-image").decode(),
                }
            )
        return {"content": content, "isError": False}

    def close(self):
        self.closed = True


class FakeMCPFactory:
    identity = {"factory": "fake-mcp"}

    def __init__(self, client):
        self.client = client

    def create(self):
        return self.client


class FakeCompletion:
    max_attempts = 1
    identity = {"transport": "fake-openrouter"}

    def __init__(self, messages):
        self.messages = list(messages)
        self.payloads = []

    @staticmethod
    def serialize(payload):
        return canonical_bytes(payload)

    def complete(self, payload):
        self.payloads.append(copy.deepcopy(payload))
        if not self.messages:
            raise AssertionError("unexpected inference request")
        message = self.messages.pop(0)
        return {
            "id": f"response-{len(self.payloads)}",
            "model": "resolved-model",
            "provider": "provider-a",
            "choices": [
                {
                    "finish_reason": "tool_calls" if message.get("tool_calls") else "stop",
                    "message": message,
                }
            ],
            "usage": {"cost": 0.01, "prompt_tokens": 10, "completion_tokens": 5},
        }


def _call(name, arguments, *, call_id="call-1"):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def _pack():
    task = standard()["tasks"][0]
    private_task = {
        "task_id": task["id"],
        "blueprint_sha256": task["blueprint_sha256"],
        "prompt": "Place one clip in the assigned project.",
        "allowed_tools": ["get_context", "place_clips"],
        "fixture_sha256": "a" * 64,
        "reference_sha256": "b" * 64,
        "verifier_sha256": "c" * 64,
    }
    pack = {
        "pack_sha256": "d" * 64,
        "tool_surface": {
            "sha256": SURFACE["tool_surface_sha256"],
            "tool_names": ["get_context", "place_clips"],
            "zero_media_cost_tools": ["get_context", "place_clips"],
        },
    }
    return private_task, pack


def _config(**overrides):
    values = {
        "model_id": "declared-model",
        "exact_model_id": "resolved-model",
        "provider_slug": "provider-a",
        "max_completion_tokens": 32,
        "max_reasoning_tokens": 32,
        "max_cost_per_task_usd": 1.0,
        "max_prompt_price_per_million": 1.0,
        "max_completion_price_per_million": 1.0,
    }
    values.update(overrides)
    return CandidateConfig(**values)


def test_candidate_calls_only_real_allowed_mcp_tools_and_finishes():
    private_task, pack = _pack()
    client = FakeMCPClient()
    completion = FakeCompletion(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    _call("place_clips", {"project_id": "project-1", "clips": []})
                ],
            },
            {"role": "assistant", "content": "Done."},
        ]
    )

    receipt = run_candidate_task(
        private_task,
        pack=pack,
        assigned_project_id="project-1",
        config=_config(),
        completion=completion,
        mcp_factory=FakeMCPFactory(client),
    )

    assert receipt["status"] == "completed"
    assert receipt["inference_calls"] == 2
    assert receipt["mcp_tool_calls"] == 1
    assert receipt["tool_call_attempts"] == 1
    assert receipt["llm_inference_cost_usd"] == 0.02
    assert receipt["paid_media_credits"] == 0.0
    assert receipt["schema_version"] == "aivideo-bench-candidate-execution-v2"
    assert receipt["token_usage"] == {
        "input_tokens": 20,
        "cached_input_tokens": 0,
        "reasoning_tokens": 0,
        "output_tokens": 10,
    }
    assert receipt["tool_metrics"][0]["name"] == "place_clips"
    assert receipt["tool_metrics"][0]["successes"] == 1
    assert receipt["first_model_response_seconds"] is not None
    assert client.calls == [
        ("place_clips", {"project_id": "project-1", "clips": []})
    ]
    assert client.rechecks == 1
    assert client.closed is True


def test_tool_outside_allowlist_is_blocked_without_mcp_dispatch():
    private_task, pack = _pack()
    client = FakeMCPClient()
    completion = FakeCompletion(
        [
            {
                "role": "assistant",
                "tool_calls": [_call("add_captions", {"project_id": "project-1"})],
            }
        ]
    )

    receipt = run_candidate_task(
        private_task,
        pack=pack,
        assigned_project_id="project-1",
        config=_config(),
        completion=completion,
        mcp_factory=FakeMCPFactory(client),
    )

    assert receipt["status"] == "tool_policy_violation"
    assert client.calls == []


def test_foreign_project_is_a_terminal_scope_violation():
    private_task, pack = _pack()
    client = FakeMCPClient()
    completion = FakeCompletion(
        [
            {
                "role": "assistant",
                "tool_calls": [
                    _call("place_clips", {"project_id": "project-foreign", "clips": []})
                ],
            }
        ]
    )

    receipt = run_candidate_task(
        private_task,
        pack=pack,
        assigned_project_id="project-1",
        config=_config(),
        completion=completion,
        mcp_factory=FakeMCPFactory(client),
    )

    assert receipt["status"] == "project_scope_violation"
    assert "foreign project" in receipt["error"]
    assert client.calls == []


def test_schema_error_is_model_visible_and_can_be_repaired():
    private_task, pack = _pack()
    client = FakeMCPClient()
    completion = FakeCompletion(
        [
            {
                "role": "assistant",
                "tool_calls": [_call("place_clips", {"project_id": "project-1", "bad": 1})],
            },
            {
                "role": "assistant",
                "tool_calls": [
                    _call(
                        "place_clips",
                        {"project_id": "project-1", "clips": []},
                        call_id="call-2",
                    )
                ],
            },
            {"role": "assistant", "content": "Repaired."},
        ]
    )

    receipt = run_candidate_task(
        private_task,
        pack=pack,
        assigned_project_id="project-1",
        config=_config(),
        completion=completion,
        mcp_factory=FakeMCPFactory(client),
    )

    assert receipt["status"] == "completed"
    assert receipt["tool_call_attempts"] == 2
    assert receipt["mcp_tool_calls"] == 1
    assert receipt["tool_metrics"] == [
        {
            "name": "place_clips",
            "attempts": 2,
            "dispatched": 1,
            "successes": 1,
            "errors": 0,
            "invalid_arguments": 1,
            "recovered_errors": 1,
            "latency_seconds": receipt["tool_metrics"][0]["latency_seconds"],
        }
    ]
    assert "invalid_tool_arguments" in completion.payloads[1]["messages"][-1]["content"]


def test_conservative_cost_preflight_blocks_before_inference():
    private_task, pack = _pack()
    client = FakeMCPClient()
    completion = FakeCompletion([])

    receipt = run_candidate_task(
        private_task,
        pack=pack,
        assigned_project_id="project-1",
        config=_config(max_cost_per_task_usd=0.0),
        completion=completion,
        mcp_factory=FakeMCPFactory(client),
    )

    assert receipt["status"] == "cost_preflight_blocked"
    assert receipt["inference_calls"] == 0
    assert completion.payloads == []


def test_mcp_media_is_forwarded_as_model_input_under_a_byte_limit():
    private_task, pack = _pack()
    client = FakeMCPClient(media=True)
    completion = FakeCompletion(
        [
            {
                "role": "assistant",
                "tool_calls": [
                    _call("place_clips", {"project_id": "project-1", "clips": []})
                ],
            },
            {"role": "assistant", "content": "Inspected."},
        ]
    )

    receipt = run_candidate_task(
        private_task,
        pack=pack,
        assigned_project_id="project-1",
        config=_config(),
        completion=completion,
        mcp_factory=FakeMCPFactory(client),
    )

    assert receipt["status"] == "completed"
    media_message = completion.payloads[1]["messages"][-1]
    assert media_message["content"][1]["type"] == "image_url"
    assert media_message["content"][1]["image_url"]["url"].startswith(
        "data:image/png;base64,"
    )


def test_live_surface_mismatch_blocks_before_inference():
    private_task, pack = _pack()
    drifted = build_tool_surface(
        protocol_version="2025-06-18",
        server_info={"name": "aivideo", "version": "changed"},
        tools=[_tool("get_context", read_only=True), _tool("place_clips")],
    )
    client = FakeMCPClient(drifted)

    with pytest.raises(Exception, match="differs from the sealed private pack"):
        run_candidate_task(
            private_task,
            pack=pack,
            assigned_project_id="project-1",
            config=_config(),
            completion=FakeCompletion([]),
            mcp_factory=FakeMCPFactory(client),
        )
    assert client.closed is True
