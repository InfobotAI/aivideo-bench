"""Private task-pack commitments for the public 100-task standard."""

from __future__ import annotations

import copy
import hashlib
import hmac
import re
from typing import Any, Mapping, Sequence

from .standard import canonical_bytes, standard


PACK_SCHEMA = "aivideo-bench-private-pack-v1"
SEAL_SCHEMA = "aivideo-bench-private-pack-hmac-v1"
SYNTHETIC_TOOL_NAMES = frozenset({"add_captions"})
TASK_KEYS = frozenset(
    {
        "task_id",
        "blueprint_sha256",
        "prompt",
        "allowed_tools",
        "fixture_sha256",
        "reference_sha256",
        "verifier_sha256",
    }
)
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _normalized_tool_names(values: Sequence[str]) -> list[str]:
    if any(not isinstance(value, str) for value in values):
        raise ValueError("MCP tool names must be strings")
    names = [value.strip() for value in values]
    if any(not name for name in names):
        raise ValueError("MCP tool names must be non-empty strings")
    if len(names) != len(set(names)):
        raise ValueError("MCP tool names must be unique")
    return sorted(names)


def _unsigned_body(pack: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(value)
        for key, value in pack.items()
        if key not in {"pack_sha256", "seal"}
    }


def _task_errors(
    tasks: Any,
    *,
    public_standard: Mapping[str, Any],
    tool_names: set[str],
    zero_media_cost_tools: set[str],
) -> list[str]:
    errors: list[str] = []
    if not isinstance(tasks, list):
        return ["private pack tasks must be a list"]
    public_tasks = {task["id"]: task for task in public_standard["tasks"]}
    seen: set[str] = set()
    ordered_ids: list[str] = []
    for index, task in enumerate(tasks, 1):
        if not isinstance(task, Mapping):
            errors.append(f"task-{index}: private task must be an object")
            continue
        task_id = str(task.get("task_id") or f"task-{index}")
        if set(task) != TASK_KEYS:
            errors.append(f"{task_id}: private task keys are not canonical")
        if task_id in seen:
            errors.append(f"{task_id}: duplicate task ID")
        seen.add(task_id)
        ordered_ids.append(task_id)
        blueprint = public_tasks.get(task_id)
        if blueprint is None:
            errors.append(f"{task_id}: unknown public task ID")
        elif task.get("blueprint_sha256") != blueprint["blueprint_sha256"]:
            errors.append(f"{task_id}: public blueprint commitment mismatch")
        prompt = task.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            errors.append(f"{task_id}: prompt must be a non-empty string")
        allowed = task.get("allowed_tools")
        if not isinstance(allowed, list) or not allowed:
            errors.append(f"{task_id}: allowed_tools must be a non-empty list")
        else:
            if any(not isinstance(name, str) for name in allowed):
                errors.append(f"{task_id}: allowed_tools must contain strings")
                continue
            if len(allowed) != len(set(allowed)):
                errors.append(f"{task_id}: allowed_tools must be unique")
            if allowed != sorted(allowed):
                errors.append(f"{task_id}: allowed_tools must be sorted canonically")
            unknown = sorted(set(allowed) - tool_names)
            if unknown:
                errors.append(f"{task_id}: tools absent from MCP snapshot: {unknown}")
            paid_or_unclassified = sorted(set(allowed) - zero_media_cost_tools)
            if paid_or_unclassified:
                errors.append(
                    f"{task_id}: tools are not classified zero-media-cost: "
                    f"{paid_or_unclassified}"
                )
            synthetic = sorted(set(allowed) & SYNTHETIC_TOOL_NAMES)
            if synthetic:
                errors.append(f"{task_id}: synthetic tools are prohibited: {synthetic}")
        for field in ("fixture_sha256", "reference_sha256", "verifier_sha256"):
            if not isinstance(task.get(field), str) or not HEX64.fullmatch(task[field]):
                errors.append(f"{task_id}: {field} must be a lowercase SHA-256")
    expected_ids = set(public_tasks)
    if ordered_ids != list(public_tasks):
        errors.append("private tasks must follow public registry order")
    missing = sorted(expected_ids - seen)
    extra = sorted(seen - expected_ids)
    if missing:
        errors.append(f"private pack is missing {len(missing)} public tasks")
    if extra:
        errors.append(f"private pack contains {len(extra)} unknown tasks")
    return errors


def seal_private_pack(
    tasks: Sequence[Mapping[str, Any]],
    *,
    tool_names: Sequence[str],
    zero_media_cost_tools: Sequence[str],
    tool_surface_sha256: str,
    key_id: str,
    signing_key: bytes,
    public_standard: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate and HMAC-seal an owner-only task pack without storing the key."""

    if len(signing_key) < 32:
        raise ValueError("private-pack signing key must contain at least 32 bytes")
    if not isinstance(key_id, str) or not key_id.strip():
        raise ValueError("private-pack key ID must be a non-empty string")
    if not isinstance(tool_surface_sha256, str) or not HEX64.fullmatch(tool_surface_sha256):
        raise ValueError("tool surface commitment must be a lowercase SHA-256")
    public = copy.deepcopy(dict(public_standard or standard()))
    names = _normalized_tool_names(tool_names)
    zero_cost_names = _normalized_tool_names(zero_media_cost_tools)
    if not set(zero_cost_names) <= set(names):
        raise ValueError("zero-media-cost tools must be present in the MCP snapshot")
    body = {
        "schema_version": PACK_SCHEMA,
        "standard_sha256": public["standard_sha256"],
        "tool_surface": {
            "sha256": tool_surface_sha256,
            "tool_names": names,
            "zero_media_cost_tools": zero_cost_names,
        },
        "tasks": [copy.deepcopy(dict(task)) for task in tasks],
    }
    errors = _task_errors(
        body["tasks"],
        public_standard=public,
        tool_names=set(names),
        zero_media_cost_tools=set(zero_cost_names),
    )
    if errors:
        raise ValueError("invalid private pack: " + "; ".join(errors))
    pack_sha256 = _sha256(body)
    signature = hmac.new(signing_key, canonical_bytes(body), hashlib.sha256).hexdigest()
    return {
        **body,
        "pack_sha256": pack_sha256,
        "seal": {
            "schema_version": SEAL_SCHEMA,
            "key_id": key_id,
            "hmac_sha256": signature,
        },
    }


def validate_private_pack(
    pack: Mapping[str, Any],
    *,
    signing_key: bytes,
    public_standard: Mapping[str, Any] | None = None,
) -> list[str]:
    """Return every structural, surface, registry, hash, and HMAC error."""

    errors: list[str] = []
    if len(signing_key) < 32:
        errors.append("private-pack signing key must contain at least 32 bytes")
    public = copy.deepcopy(dict(public_standard or standard()))
    body = _unsigned_body(pack)
    if body.get("schema_version") != PACK_SCHEMA:
        errors.append("private pack schema mismatch")
    if body.get("standard_sha256") != public.get("standard_sha256"):
        errors.append("private pack standard commitment mismatch")
    surface = body.get("tool_surface")
    zero_cost_names: list[str] = []
    if not isinstance(surface, Mapping):
        errors.append("private pack tool surface must be an object")
        names: list[str] = []
    else:
        raw_names = surface.get("tool_names")
        raw_zero_cost = surface.get("zero_media_cost_tools")
        try:
            names = _normalized_tool_names(raw_names if isinstance(raw_names, list) else [])
        except ValueError as error:
            errors.append(str(error))
            names = []
        if raw_names != names:
            errors.append("private pack tool names must be sorted canonically")
        try:
            zero_cost_names = _normalized_tool_names(
                raw_zero_cost if isinstance(raw_zero_cost, list) else []
            )
        except ValueError as error:
            errors.append(str(error))
            zero_cost_names = []
        if raw_zero_cost != zero_cost_names:
            errors.append("zero-media-cost tools must be sorted canonically")
        if not set(zero_cost_names) <= set(names):
            errors.append("zero-media-cost tools are absent from the MCP snapshot")
        if not isinstance(surface.get("sha256"), str) or not HEX64.fullmatch(
            surface["sha256"]
        ):
            errors.append("private pack tool surface hash is malformed")
        synthetic = sorted(set(names) & SYNTHETIC_TOOL_NAMES)
        if synthetic:
            errors.append(f"MCP snapshot contains synthetic tools: {synthetic}")
    errors.extend(
        _task_errors(
            body.get("tasks"),
            public_standard=public,
            tool_names=set(names),
            zero_media_cost_tools=set(zero_cost_names),
        )
    )
    if pack.get("pack_sha256") != _sha256(body):
        errors.append("private pack hash mismatch")
    seal = pack.get("seal")
    if not isinstance(seal, Mapping):
        errors.append("private pack seal must be an object")
    else:
        if seal.get("schema_version") != SEAL_SCHEMA:
            errors.append("private pack seal schema mismatch")
        supplied = seal.get("hmac_sha256")
        expected = hmac.new(signing_key, canonical_bytes(body), hashlib.sha256).hexdigest()
        if not isinstance(supplied, str) or not hmac.compare_digest(supplied, expected):
            errors.append("private pack HMAC mismatch")
        if not isinstance(seal.get("key_id"), str) or not seal["key_id"].strip():
            errors.append("private pack key ID is malformed")
    return errors
