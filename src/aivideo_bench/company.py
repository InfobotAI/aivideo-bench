"""Company-facing model comparison without inventing a composite business score."""

from __future__ import annotations

import math
import hashlib
import hmac
from collections import defaultdict
from statistics import mean
from typing import Any, Mapping, Sequence

from .results import summarize_results
from .standard import canonical_bytes, sha256, standard


COMPANY_INPUT_SCHEMA_VERSION = "aivideo-bench-company-candidate-v2"
COMPANY_REPORT_SCHEMA_VERSION = "aivideo-bench-company-report-v2"
RUN_MANIFEST_SCHEMA_VERSION = "aivideo-bench-run-manifest-v1"
CLOCK_TOLERANCE_SECONDS = 0.00001
CLAIM_LEVELS = ("observed", "directional", "causal")
BUSINESS_SOURCES = (
    "production_telemetry",
    "billing_ledger",
    "support_ledger",
    "experiment",
    "sales_review",
    "manual_audit",
)

# The registry is the horizontal extension point. A new company question is one
# data entry; validation and reporting remain unchanged.
BUSINESS_METRICS: tuple[dict[str, str], ...] = (
    {
        "id": "paid_agent_task_start_rate",
        "label": "Paid agent task start rate",
        "question": "Do paid users choose the agent for real work?",
        "unit": "rate",
        "direction": "higher",
        "source": "production_telemetry",
    },
    {
        "id": "task_to_usable_project_rate",
        "label": "Task to usable project",
        "question": "Does an agent task leave a project the user can continue using?",
        "unit": "rate",
        "direction": "higher",
        "source": "production_telemetry",
    },
    {
        "id": "task_to_export_rate",
        "label": "Task to export",
        "question": "How often does an agent task lead to a successful export?",
        "unit": "rate",
        "direction": "higher",
        "source": "production_telemetry",
    },
    {
        "id": "generation_conversion_rate",
        "label": "Generation conversion",
        "question": "How often does agent work lead to an intentional media generation?",
        "unit": "rate",
        "direction": "higher",
        "source": "production_telemetry",
    },
    {
        "id": "support_contacts_per_100_tasks",
        "label": "Support contacts per 100 tasks",
        "question": "How much support burden follows agent use?",
        "unit": "per_100",
        "direction": "lower",
        "source": "support_ledger",
    },
    {
        "id": "refund_rate_after_agent_task",
        "label": "Refund rate after agent task",
        "question": "Is agent use associated with refunds?",
        "unit": "rate",
        "direction": "lower",
        "source": "billing_ledger",
    },
    {
        "id": "topup_conversion_rate",
        "label": "Credit top-up conversion",
        "question": "Does successful agent use lead to credit top-ups?",
        "unit": "rate",
        "direction": "higher",
        "source": "billing_ledger",
    },
    {
        "id": "upgrade_conversion_rate",
        "label": "Plan upgrade conversion",
        "question": "Does successful agent use lead to plan upgrades?",
        "unit": "rate",
        "direction": "higher",
        "source": "billing_ledger",
    },
    {
        "id": "seven_day_return_rate",
        "label": "7-day return rate",
        "question": "Do agent users return within seven days?",
        "unit": "rate",
        "direction": "higher",
        "source": "production_telemetry",
    },
    {
        "id": "thirty_day_retention_rate",
        "label": "30-day retention rate",
        "question": "Are agent users retained after thirty days?",
        "unit": "rate",
        "direction": "higher",
        "source": "production_telemetry",
    },
    {
        "id": "cancellation_rate_after_agent_task",
        "label": "Cancellation rate after agent task",
        "question": "Is agent use associated with subscription cancellation?",
        "unit": "rate",
        "direction": "lower",
        "source": "billing_ledger",
    },
    {
        "id": "credit_screen_abandonment_rate",
        "label": "Credit-screen abandonment",
        "question": "Do users abandon work when the agent encounters a credit screen?",
        "unit": "rate",
        "direction": "lower",
        "source": "production_telemetry",
    },
    {
        "id": "customer_rework_rate",
        "label": "Customer rework rate",
        "question": "How often must users substantially repair agent work?",
        "unit": "rate",
        "direction": "lower",
        "source": "manual_audit",
    },
    {
        "id": "project_loss_rate",
        "label": "Project loss rate",
        "question": "How often does agent use destroy or irrecoverably damage user work?",
        "unit": "rate",
        "direction": "lower",
        "source": "support_ledger",
    },
    {
        "id": "sales_demo_success_rate",
        "label": "Sales demo success rate",
        "question": "Can the agent reliably complete the promised live demo workflow?",
        "unit": "rate",
        "direction": "higher",
        "source": "sales_review",
    },
    {
        "id": "revenue_per_agent_task_usd",
        "label": "Revenue per agent task",
        "question": "How much attributable revenue is observed per agent task?",
        "unit": "usd",
        "direction": "higher",
        "source": "billing_ledger",
    },
    {
        "id": "gross_margin_per_agent_task_usd",
        "label": "Gross margin per agent task",
        "question": "Does agent use create value after inference and media costs?",
        "unit": "signed_usd",
        "direction": "higher",
        "source": "billing_ledger",
    },
)

BUSINESS_METRIC_BY_ID = {metric["id"]: metric for metric in BUSINESS_METRICS}

RUN_ROW_KEYS = frozenset(
    {
        "task_id",
        "trial",
        "receipt_sha256",
        "pack_sha256",
        "tool_surface_sha256",
        "candidate_config_sha256",
        "prompt_policy_sha256",
        "verifier_policy_version",
        "verifier_policy_sha256",
        "verifier_evidence_sha256",
        "status",
        "latency_seconds",
        "first_model_response_seconds",
        "inference_seconds",
        "mcp_seconds",
        "input_tokens",
        "cached_input_tokens",
        "reasoning_tokens",
        "output_tokens",
        "inference_calls",
        "llm_inference_cost_usd",
        "paid_media_credits",
        "cost_cap_respected",
        "tool_metrics",
        "human_interventions",
        "false_completion",
        "destructive_side_effect",
        "context_limit_hit",
        "stuck_loop",
        "external_dependency_failure",
    }
)
TOOL_METRIC_KEYS = frozenset(
    {
        "name",
        "attempts",
        "dispatched",
        "successes",
        "errors",
        "invalid_arguments",
        "redundant_calls",
        "same_tool_successes_after_error",
        "latency_seconds",
    }
)
BUSINESS_OBSERVATION_KEYS = frozenset(
    {
        "metric_id",
        "value",
        "numerator",
        "denominator",
        "window",
        "claim_level",
        "source",
        "evidence",
    }
)
RUN_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "variant_id",
        "comparison_kind",
        "ablation_group_id",
        "standard_version",
        "standard_sha256",
        "task_pack_identity_sha256",
        "pack_sha256",
        "tool_surface_sha256",
        "candidate_config",
        "candidate_config_sha256",
        "prompt_policy_sha256",
        "verifier_policy_version",
        "verifier_policy_sha256",
        "calibration_gate",
        "result_matrix_sha256",
        "runtime_matrix_sha256",
        "result_binding_sha256",
        "business_observations_sha256",
        "attestation",
    }
)
CANDIDATE_KEYS = frozenset(
    {
        "schema_version",
        "manifest",
        "result_rows",
        "run_rows",
        "result_receipt_bindings",
        "business_observations",
    }
)
RESULT_BINDING_KEYS = frozenset({"task_id", "trial", "receipt_sha256"})
VERIFIER_ANNOTATION_KEYS = frozenset(
    {
        "verifier_policy_version",
        "verifier_policy_sha256",
        "verifier_evidence_sha256",
        "human_interventions",
        "false_completion",
        "destructive_side_effect",
        "context_limit_hit",
        "stuck_loop",
        "external_dependency_failure",
        "redundant_calls_by_tool",
    }
)
OBSERVED_EVIDENCE_KEYS = frozenset(
    {"cohort_id", "query_sha256", "exposure_identity_sha256"}
)
DIRECTIONAL_EVIDENCE_KEYS = frozenset(
    {
        "cohort_id",
        "query_sha256",
        "exposure_identity_sha256",
        "comparator_numerator",
        "comparator_denominator",
        "comparator_window",
        "comparator_query_sha256",
        "comparator_exposure_identity_sha256",
    }
)
CAUSAL_EVIDENCE_KEYS = frozenset(
    {
        "experiment_id",
        "assignment_unit",
        "control_numerator",
        "control_denominator",
        "analysis_policy_version",
        "analysis_artifact_sha256",
        "exposure_identity_sha256",
        "control_exposure_identity_sha256",
        "effect_ci_lower",
        "effect_ci_upper",
    }
)
CALIBRATION_GATE_KEYS = frozenset(
    {"policy_version", "evidence_sha256", "passed"}
)
ATTESTATION_KEYS = frozenset({"key_id", "algorithm", "hmac_sha256"})
RUNTIME_STATUSES = frozenset(
    {
        "completed",
        "request_too_large",
        "cost_preflight_blocked",
        "cost_cap_exceeded",
        "tool_limit_exceeded",
        "tool_policy_violation",
        "project_scope_violation",
        "step_limit_exceeded",
        "harness_error",
    }
)
EXECUTION_INVALID_STATUSES = frozenset(
    {"tool_policy_violation", "project_scope_violation", "harness_error"}
)
COST_GATE_FAILED_STATUSES = frozenset(
    {"cost_preflight_blocked", "cost_cap_exceeded"}
)


def metric_catalog() -> dict[str, Any]:
    """Return the stable, public catalog of business questions."""

    return {
        "schema_version": COMPANY_REPORT_SCHEMA_VERSION,
        "principle": "quality, operating economics, and business outcomes stay separate",
        "claim_levels": list(CLAIM_LEVELS),
        "metrics": [dict(metric) for metric in BUSINESS_METRICS],
    }


def _number(
    value: Any, label: str, *, minimum: float = 0.0, maximum: float | None = None
) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric, not boolean")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be numeric") from None
    if not math.isfinite(number) or number < minimum or (
        maximum is not None and number > maximum
    ):
        ceiling = "infinity" if maximum is None else str(maximum)
        raise ValueError(f"{label} must be in [{minimum},{ceiling}]")
    return number


def _optional_number(value: Any, label: str) -> float | None:
    return None if value is None else _number(value, label)


def _count(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _nonempty(value: Any, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{label} must be non-empty")
    return text


def _sha256_value(value: Any, label: str) -> str:
    digest = _nonempty(value, label)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return digest


def _matrix_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    ordered = sorted(rows, key=lambda row: (str(row.get("task_id")), row.get("trial")))
    return sha256(ordered)


def build_run_manifest(
    *,
    variant_id: str,
    comparison_kind: str,
    ablation_group_id: str | None,
    task_pack_identity_sha256: str,
    pack_sha256: str,
    tool_surface_sha256: str,
    candidate_config: Mapping[str, Any],
    prompt_policy_sha256: str,
    verifier_policy_version: str,
    verifier_policy_sha256: str,
    calibration_gate: Mapping[str, Any],
    result_rows: Sequence[Mapping[str, Any]],
    run_rows: Sequence[Mapping[str, Any]],
    result_receipt_bindings: Sequence[Mapping[str, Any]],
    business_observations: Sequence[Mapping[str, Any]],
    key_id: str,
    signing_key: bytes,
    public_standard: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build and authenticate the identity binding quality and runtime matrices."""

    public = dict(public_standard or standard())
    if comparison_kind not in {"standard", "tool_ablation"}:
        raise ValueError("comparison_kind must be standard or tool_ablation")
    if comparison_kind == "tool_ablation" and not ablation_group_id:
        raise ValueError("tool ablation requires ablation_group_id")
    if not isinstance(candidate_config, Mapping):
        raise ValueError("candidate_config must be an object")
    config = dict(candidate_config)
    _nonempty(config.get("exact_model_id"), "candidate_config.exact_model_id")
    _nonempty(config.get("provider_slug"), "candidate_config.provider_slug")
    if set(calibration_gate) != CALIBRATION_GATE_KEYS:
        raise ValueError("calibration gate keys are not canonical")
    if not isinstance(calibration_gate["passed"], bool):
        raise ValueError("calibration gate passed must be boolean")
    if not isinstance(signing_key, bytes) or len(signing_key) < 16:
        raise ValueError("manifest signing key must contain at least 16 bytes")
    body = {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "variant_id": _nonempty(variant_id, "variant_id"),
        "comparison_kind": comparison_kind,
        "ablation_group_id": (
            None
            if ablation_group_id is None
            else _nonempty(ablation_group_id, "ablation_group_id")
        ),
        "standard_version": public["standard_version"],
        "standard_sha256": public["standard_sha256"],
        "task_pack_identity_sha256": _sha256_value(
            task_pack_identity_sha256, "task_pack_identity_sha256"
        ),
        "pack_sha256": _sha256_value(pack_sha256, "pack_sha256"),
        "tool_surface_sha256": _sha256_value(
            tool_surface_sha256, "tool_surface_sha256"
        ),
        "candidate_config": config,
        "candidate_config_sha256": sha256(config),
        "prompt_policy_sha256": _sha256_value(
            prompt_policy_sha256, "prompt_policy_sha256"
        ),
        "verifier_policy_version": _nonempty(
            verifier_policy_version, "verifier_policy_version"
        ),
        "verifier_policy_sha256": _sha256_value(
            verifier_policy_sha256, "verifier_policy_sha256"
        ),
        "calibration_gate": {
            "policy_version": _nonempty(
                calibration_gate["policy_version"], "calibration_gate.policy_version"
            ),
            "evidence_sha256": _sha256_value(
                calibration_gate["evidence_sha256"],
                "calibration_gate.evidence_sha256",
            ),
            "passed": calibration_gate["passed"],
        },
        "result_matrix_sha256": _matrix_sha256(result_rows),
        "runtime_matrix_sha256": _matrix_sha256(run_rows),
        "result_binding_sha256": _matrix_sha256(result_receipt_bindings),
        "business_observations_sha256": sha256(
            sorted(business_observations, key=lambda row: str(row.get("metric_id")))
        ),
    }
    signature = hmac.new(signing_key, canonical_bytes(body), hashlib.sha256).hexdigest()
    return {
        **body,
        "attestation": {
            "key_id": _nonempty(key_id, "key_id"),
            "algorithm": "hmac-sha256",
            "hmac_sha256": signature,
        },
    }


def _rate(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0 else round(numerator / denominator, 6)


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile requires at least one value")
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _expected_cells(public: Mapping[str, Any]) -> set[tuple[str, int]]:
    return {
        (task["id"], trial)
        for task in public["tasks"]
        for trial in range(1, int(public["policy"]["trials_per_task"]) + 1)
    }


def _validate_tool_metric(raw: Mapping[str, Any], label: str) -> dict[str, Any]:
    if set(raw) != TOOL_METRIC_KEYS:
        raise ValueError(f"{label} keys are not canonical")
    name = str(raw["name"]).strip()
    if not name:
        raise ValueError(f"{label}.name must be non-empty")
    counts = {
        key: _count(raw[key], f"{label}.{key}")
        for key in (
            "attempts",
            "dispatched",
            "successes",
            "errors",
            "invalid_arguments",
            "redundant_calls",
            "same_tool_successes_after_error",
        )
    }
    if counts["successes"] + counts["errors"] != counts["dispatched"]:
        raise ValueError(f"{label}: successes plus errors must equal dispatched")
    if counts["dispatched"] > counts["attempts"]:
        raise ValueError(f"{label}: dispatched cannot exceed attempts")
    if counts["invalid_arguments"] > counts["attempts"] - counts["dispatched"]:
        raise ValueError(f"{label}: invalid arguments must be undispatched attempts")
    if counts["redundant_calls"] > counts["dispatched"]:
        raise ValueError(f"{label}: redundant calls cannot exceed dispatched")
    preceding_errors = counts["invalid_arguments"] + counts["errors"]
    if counts["same_tool_successes_after_error"] > preceding_errors:
        raise ValueError(
            f"{label}: same-tool successes after error exceed observed errors"
        )
    return {
        "name": name,
        **counts,
        "latency_seconds": _number(raw["latency_seconds"], f"{label}.latency_seconds"),
    }


def _validate_run_rows(
    rows: Sequence[Mapping[str, Any]],
    public: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    expected = _expected_cells(public)
    seen: set[tuple[str, int]] = set()
    validated: list[dict[str, Any]] = []
    for index, raw in enumerate(rows, 1):
        label = f"run-row-{index}"
        if set(raw) != RUN_ROW_KEYS:
            raise ValueError(f"{label}: keys are not canonical")
        task_id = str(raw["task_id"])
        status = str(raw["status"]).strip()
        if status not in RUNTIME_STATUSES:
            raise ValueError(f"{label}.status is not canonical")
        trial = raw["trial"]
        if isinstance(trial, bool) or not isinstance(trial, int):
            raise ValueError(f"{label}.trial must be an integer")
        cell = (task_id, trial)
        if cell not in expected:
            raise ValueError(f"{label}: unknown result cell {cell}")
        if cell in seen:
            raise ValueError(f"{label}: duplicate result cell {cell}")
        seen.add(cell)
        booleans = {
            key: raw[key]
            for key in (
                "false_completion",
                "destructive_side_effect",
                "context_limit_hit",
                "stuck_loop",
                "external_dependency_failure",
            )
        }
        if any(not isinstance(value, bool) for value in booleans.values()):
            raise ValueError(f"{label}: incident fields must be booleans")
        if not isinstance(raw["cost_cap_respected"], bool):
            raise ValueError(f"{label}.cost_cap_respected must be boolean")
        tool_metrics = raw["tool_metrics"]
        if not isinstance(tool_metrics, list):
            raise ValueError(f"{label}.tool_metrics must be a list")
        tools = [
            _validate_tool_metric(tool, f"{label}.tool_metrics[{tool_index}]")
            for tool_index, tool in enumerate(tool_metrics)
        ]
        names = [tool["name"] for tool in tools]
        if len(names) != len(set(names)) or names != sorted(names):
            raise ValueError(f"{label}.tool_metrics must have unique canonical names")
        validated.append(
            {
                "task_id": task_id,
                "trial": trial,
                "status": status,
                **{
                    key: _sha256_value(raw[key], f"{label}.{key}")
                    for key in (
                        "receipt_sha256",
                        "pack_sha256",
                        "tool_surface_sha256",
                        "candidate_config_sha256",
                        "prompt_policy_sha256",
                        "verifier_policy_sha256",
                        "verifier_evidence_sha256",
                    )
                },
                "verifier_policy_version": _nonempty(
                    raw["verifier_policy_version"],
                    f"{label}.verifier_policy_version",
                ),
                **{
                    key: _number(raw[key], f"{label}.{key}")
                    for key in (
                        "latency_seconds",
                        "inference_seconds",
                        "mcp_seconds",
                        "llm_inference_cost_usd",
                        "paid_media_credits",
                    )
                },
                "first_model_response_seconds": _optional_number(
                    raw["first_model_response_seconds"],
                    f"{label}.first_model_response_seconds",
                ),
                **{
                    key: _count(raw[key], f"{label}.{key}")
                    for key in (
                        "input_tokens",
                        "cached_input_tokens",
                        "reasoning_tokens",
                        "output_tokens",
                        "inference_calls",
                        "human_interventions",
                    )
                },
                "tool_metrics": tools,
                "cost_cap_respected": raw["cost_cap_respected"],
                **booleans,
            }
        )
        row = validated[-1]
        for key in (
            "pack_sha256",
            "tool_surface_sha256",
            "candidate_config_sha256",
            "prompt_policy_sha256",
            "verifier_policy_version",
            "verifier_policy_sha256",
        ):
            if row[key] != manifest[key]:
                raise ValueError(f"{label}: {key} differs from run manifest")
        if (
            row["first_model_response_seconds"] is not None
            and row["first_model_response_seconds"]
            > row["latency_seconds"] + CLOCK_TOLERANCE_SECONDS
        ):
            raise ValueError(f"{label}: first model response exceeds total latency")
        for key in ("inference_seconds", "mcp_seconds"):
            if row[key] > row["latency_seconds"] + CLOCK_TOLERANCE_SECONDS:
                raise ValueError(f"{label}: {key} exceeds total latency")
        if (
            row["inference_seconds"] + row["mcp_seconds"]
            > row["latency_seconds"] + CLOCK_TOLERANCE_SECONDS
        ):
            raise ValueError(f"{label}: inference plus MCP time exceeds total latency")
        tool_latency = sum(tool["latency_seconds"] for tool in row["tool_metrics"])
        if not math.isclose(
            tool_latency,
            row["mcp_seconds"],
            rel_tol=0.0,
            abs_tol=CLOCK_TOLERANCE_SECONDS,
        ):
            raise ValueError(f"{label}: per-tool latency differs from MCP time")
        if row["cached_input_tokens"] > row["input_tokens"]:
            raise ValueError(f"{label}: cached input tokens exceed input tokens")
    if seen != expected:
        raise ValueError(
            "run telemetry must contain exactly 300 canonical cells "
            f"(missing={len(expected - seen)}, extra={len(seen - expected)})"
        )
    return validated


def _validate_result_bindings(
    rows: Sequence[Mapping[str, Any]],
    *,
    public: Mapping[str, Any],
    runs: Sequence[Mapping[str, Any]],
) -> None:
    expected = _expected_cells(public)
    run_receipts = {
        (row["task_id"], row["trial"]): row["receipt_sha256"] for row in runs
    }
    seen: set[tuple[str, int]] = set()
    for index, raw in enumerate(rows, 1):
        label = f"result-binding-{index}"
        if set(raw) != RESULT_BINDING_KEYS:
            raise ValueError(f"{label}: keys are not canonical")
        trial = raw["trial"]
        if isinstance(trial, bool) or not isinstance(trial, int):
            raise ValueError(f"{label}.trial must be an integer")
        cell = (str(raw["task_id"]), trial)
        if cell not in expected:
            raise ValueError(f"{label}: unknown result cell {cell}")
        if cell in seen:
            raise ValueError(f"{label}: duplicate result cell {cell}")
        seen.add(cell)
        receipt_sha = _sha256_value(raw["receipt_sha256"], f"{label}.receipt_sha256")
        if receipt_sha != run_receipts[cell]:
            raise ValueError(f"{label}: result and runtime receipt identities differ")
    if seen != expected:
        raise ValueError(
            "result bindings must contain exactly 300 canonical cells "
            f"(missing={len(expected - seen)}, extra={len(seen - expected)})"
        )


def run_row_from_receipt(
    receipt: Mapping[str, Any],
    *,
    trial: int,
    annotations: Mapping[str, Any],
) -> dict[str, Any]:
    """Convert a v2 execution receipt plus verifier annotations to telemetry."""

    if receipt.get("schema_version") != "aivideo-bench-candidate-execution-v2":
        raise ValueError("company telemetry requires a v2 candidate execution receipt")
    notes = dict(annotations)
    if set(notes) != VERIFIER_ANNOTATION_KEYS:
        raise ValueError("verifier annotation keys are not complete and canonical")
    redundant = notes["redundant_calls_by_tool"]
    if not isinstance(redundant, Mapping):
        raise ValueError("redundant_calls_by_tool must be an object")
    token_usage = receipt.get("token_usage")
    receipt_tools = receipt.get("tool_metrics")
    candidate_config = receipt.get("candidate")
    if (
        not isinstance(token_usage, Mapping)
        or not isinstance(receipt_tools, list)
        or not isinstance(candidate_config, Mapping)
    ):
        raise ValueError("v2 receipt is missing candidate, token, or tool telemetry")
    if any(not isinstance(tool, Mapping) for tool in receipt_tools):
        raise ValueError("receipt tool metrics must be objects")
    receipt_tool_names = {str(tool.get("name") or "") for tool in receipt_tools}
    if set(redundant) != receipt_tool_names:
        raise ValueError("redundant tool annotations must exactly match receipt tools")
    tools = []
    for index, tool in enumerate(receipt_tools):
        if not isinstance(tool, Mapping):
            raise ValueError(f"receipt tool metric {index} must be an object")
        name = str(tool.get("name") or "")
        tools.append(
            {
                "name": name,
                "attempts": tool.get("attempts"),
                "dispatched": tool.get("dispatched"),
                "successes": tool.get("successes"),
                "errors": tool.get("errors"),
                "invalid_arguments": tool.get("invalid_arguments"),
                "redundant_calls": redundant.get(name, 0),
                "same_tool_successes_after_error": tool.get(
                    "same_tool_successes_after_error"
                ),
                "latency_seconds": tool.get("latency_seconds"),
            }
        )
    row = {
        "task_id": receipt.get("task_id"),
        "trial": trial,
        "receipt_sha256": sha256(receipt),
        "pack_sha256": receipt.get("pack_sha256"),
        "tool_surface_sha256": receipt.get("tool_surface_sha256"),
        "candidate_config_sha256": sha256(dict(candidate_config)),
        "prompt_policy_sha256": receipt.get("prompt_policy_sha256"),
        "verifier_policy_version": notes["verifier_policy_version"],
        "verifier_policy_sha256": notes["verifier_policy_sha256"],
        "verifier_evidence_sha256": notes["verifier_evidence_sha256"],
        "status": receipt.get("status"),
        "latency_seconds": receipt.get("latency_seconds"),
        "first_model_response_seconds": receipt.get("first_model_response_seconds"),
        "inference_seconds": receipt.get("inference_seconds"),
        "mcp_seconds": receipt.get("mcp_seconds"),
        "input_tokens": token_usage.get("input_tokens"),
        "cached_input_tokens": token_usage.get("cached_input_tokens"),
        "reasoning_tokens": token_usage.get("reasoning_tokens"),
        "output_tokens": token_usage.get("output_tokens"),
        "inference_calls": receipt.get("inference_calls"),
        "llm_inference_cost_usd": receipt.get("llm_inference_cost_usd"),
        "paid_media_credits": receipt.get("paid_media_credits"),
        "cost_cap_respected": receipt.get("cost_cap_respected"),
        "tool_metrics": tools,
        "human_interventions": notes["human_interventions"],
        "false_completion": notes["false_completion"],
        "destructive_side_effect": notes["destructive_side_effect"],
        "context_limit_hit": notes["context_limit_hit"],
        "stuck_loop": notes["stuck_loop"],
        "external_dependency_failure": notes["external_dependency_failure"],
    }
    # Validate one row through the same primitives; matrix completeness is
    # intentionally enforced only after all 300 rows are assembled.
    if isinstance(row["trial"], bool) or not isinstance(row["trial"], int):
        raise ValueError("trial must be an integer")
    for index, tool in enumerate(row["tool_metrics"]):
        _validate_tool_metric(tool, f"tool_metrics[{index}]")
    return row


def _business_observations(
    rows: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    observations: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(rows, 1):
        label = f"business-observation-{index}"
        if set(raw) != BUSINESS_OBSERVATION_KEYS:
            raise ValueError(f"{label}: keys are not canonical")
        metric_id = str(raw["metric_id"])
        if metric_id not in BUSINESS_METRIC_BY_ID:
            raise ValueError(f"{label}: unknown metric ID {metric_id}")
        if metric_id in observations:
            raise ValueError(f"{label}: duplicate metric ID {metric_id}")
        metric = BUSINESS_METRIC_BY_ID[metric_id]
        maximum = 1.0 if metric["unit"] == "rate" else None
        minimum = -math.inf if metric["unit"] == "signed_usd" else 0.0
        value = _number(
            raw["value"], f"{label}.value", minimum=minimum, maximum=maximum
        )
        numerator = _number(raw["numerator"], f"{label}.numerator", minimum=minimum)
        denominator = _count(raw["denominator"], f"{label}.denominator")
        if denominator == 0:
            raise ValueError(f"{label}.denominator must be positive")
        claim_level = str(raw["claim_level"])
        source = str(raw["source"])
        window = str(raw["window"]).strip()
        if claim_level not in CLAIM_LEVELS:
            raise ValueError(f"{label}: unsupported claim level {claim_level}")
        if source not in BUSINESS_SOURCES:
            raise ValueError(f"{label}: unsupported source {source}")
        if not window:
            raise ValueError(f"{label}.window must be non-empty")
        expected_value = numerator / denominator
        if metric["unit"] == "per_100":
            expected_value *= 100
        if not math.isclose(value, expected_value, rel_tol=1e-6, abs_tol=1e-9):
            raise ValueError(f"{label}: value does not match numerator and denominator")
        evidence = raw["evidence"]
        if not isinstance(evidence, Mapping):
            raise ValueError(f"{label}.evidence must be an object")
        expected_keys = {
            "observed": OBSERVED_EVIDENCE_KEYS,
            "directional": DIRECTIONAL_EVIDENCE_KEYS,
            "causal": CAUSAL_EVIDENCE_KEYS,
        }[claim_level]
        if set(evidence) != expected_keys:
            raise ValueError(f"{label}: {claim_level} evidence keys are not canonical")
        if evidence.get("exposure_identity_sha256") != manifest["candidate_config_sha256"]:
            raise ValueError(f"{label}: business exposure differs from candidate identity")
        validated_evidence: dict[str, Any]
        if claim_level == "observed":
            if source != metric["source"]:
                raise ValueError(f"{label}: observed source differs from metric registry")
            validated_evidence = {
                "cohort_id": _nonempty(evidence["cohort_id"], f"{label}.cohort_id"),
                "query_sha256": _sha256_value(
                    evidence["query_sha256"], f"{label}.query_sha256"
                ),
                "exposure_identity_sha256": manifest["candidate_config_sha256"],
            }
        elif claim_level == "directional":
            if source != metric["source"]:
                raise ValueError(f"{label}: directional source differs from metric registry")
            comparator_numerator = _number(
                evidence["comparator_numerator"],
                f"{label}.comparator_numerator",
                minimum=minimum,
            )
            comparator_denominator = _count(
                evidence["comparator_denominator"],
                f"{label}.comparator_denominator",
            )
            if comparator_denominator == 0:
                raise ValueError(f"{label}.comparator_denominator must be positive")
            comparator_value = comparator_numerator / comparator_denominator
            if metric["unit"] == "per_100":
                comparator_value *= 100
            if metric["unit"] == "rate" and comparator_value > 1:
                raise ValueError(f"{label}: comparator rate exceeds 1")
            validated_evidence = {
                "cohort_id": _nonempty(evidence["cohort_id"], f"{label}.cohort_id"),
                "query_sha256": _sha256_value(
                    evidence["query_sha256"], f"{label}.query_sha256"
                ),
                "exposure_identity_sha256": manifest["candidate_config_sha256"],
                "comparator_numerator": comparator_numerator,
                "comparator_denominator": comparator_denominator,
                "comparator_window": _nonempty(
                    evidence["comparator_window"], f"{label}.comparator_window"
                ),
                "comparator_query_sha256": _sha256_value(
                    evidence["comparator_query_sha256"],
                    f"{label}.comparator_query_sha256",
                ),
                "comparator_exposure_identity_sha256": _sha256_value(
                    evidence["comparator_exposure_identity_sha256"],
                    f"{label}.comparator_exposure_identity_sha256",
                ),
                "comparator_value": round(comparator_value, 9),
                "absolute_delta": round(value - comparator_value, 9),
            }
        else:
            if source != "experiment":
                raise ValueError(f"{label}: causal claims require experiment source")
            control_numerator = _number(
                evidence["control_numerator"],
                f"{label}.control_numerator",
                minimum=minimum,
            )
            control_denominator = _count(
                evidence["control_denominator"], f"{label}.control_denominator"
            )
            if control_denominator == 0:
                raise ValueError(f"{label}.control_denominator must be positive")
            assignment_unit = _nonempty(
                evidence["assignment_unit"], f"{label}.assignment_unit"
            )
            if assignment_unit not in {"user", "project", "session", "task"}:
                raise ValueError(f"{label}: unsupported experiment assignment unit")
            control_value = control_numerator / control_denominator
            if metric["unit"] == "per_100":
                control_value *= 100
            if metric["unit"] == "rate" and control_value > 1:
                raise ValueError(f"{label}: control rate exceeds 1")
            effect = value - control_value
            effect_minimum = -1.0 if metric["unit"] == "rate" else -math.inf
            effect_maximum = 1.0 if metric["unit"] == "rate" else None
            ci_lower = _number(
                evidence["effect_ci_lower"],
                f"{label}.effect_ci_lower",
                minimum=effect_minimum,
                maximum=effect_maximum,
            )
            ci_upper = _number(
                evidence["effect_ci_upper"],
                f"{label}.effect_ci_upper",
                minimum=effect_minimum,
                maximum=effect_maximum,
            )
            if ci_lower > effect or effect > ci_upper:
                raise ValueError(f"{label}: effect is outside confidence interval")
            validated_evidence = {
                "experiment_id": _nonempty(
                    evidence["experiment_id"], f"{label}.experiment_id"
                ),
                "assignment_unit": assignment_unit,
                "control_numerator": control_numerator,
                "control_denominator": control_denominator,
                "analysis_policy_version": _nonempty(
                    evidence["analysis_policy_version"],
                    f"{label}.analysis_policy_version",
                ),
                "analysis_artifact_sha256": _sha256_value(
                    evidence["analysis_artifact_sha256"],
                    f"{label}.analysis_artifact_sha256",
                ),
                "exposure_identity_sha256": manifest["candidate_config_sha256"],
                "control_exposure_identity_sha256": _sha256_value(
                    evidence["control_exposure_identity_sha256"],
                    f"{label}.control_exposure_identity_sha256",
                ),
                "control_value": round(control_value, 9),
                "absolute_effect": round(effect, 9),
                "effect_ci_lower": ci_lower,
                "effect_ci_upper": ci_upper,
            }
        observations[metric_id] = {
            "value": value,
            "numerator": numerator,
            "denominator": denominator,
            "window": window,
            "claim_level": claim_level,
            "source": source,
            "evidence": validated_evidence,
        }
    return observations


def _validate_run_manifest(
    raw: Mapping[str, Any],
    *,
    signing_key: bytes,
    public: Mapping[str, Any],
    result_rows: Sequence[Mapping[str, Any]],
    run_rows: Sequence[Mapping[str, Any]],
    result_receipt_bindings: Sequence[Mapping[str, Any]],
    business_observations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if set(raw) != RUN_MANIFEST_KEYS:
        raise ValueError("run manifest keys are not canonical")
    if raw.get("schema_version") != RUN_MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported run manifest schema_version")
    if not isinstance(signing_key, bytes) or len(signing_key) < 16:
        raise ValueError("manifest signing key must contain at least 16 bytes")
    attestation = raw["attestation"]
    if not isinstance(attestation, Mapping) or set(attestation) != ATTESTATION_KEYS:
        raise ValueError("manifest attestation keys are not canonical")
    if attestation.get("algorithm") != "hmac-sha256":
        raise ValueError("unsupported manifest attestation algorithm")
    _nonempty(attestation.get("key_id"), "manifest.attestation.key_id")
    supplied_hmac = _sha256_value(
        attestation.get("hmac_sha256"), "manifest.attestation.hmac_sha256"
    )
    body = {key: value for key, value in raw.items() if key != "attestation"}
    expected_hmac = hmac.new(
        signing_key, canonical_bytes(body), hashlib.sha256
    ).hexdigest()
    if not hmac.compare_digest(supplied_hmac, expected_hmac):
        raise ValueError("run manifest HMAC mismatch")
    comparison_kind = str(raw["comparison_kind"])
    if comparison_kind not in {"standard", "tool_ablation"}:
        raise ValueError("comparison_kind must be standard or tool_ablation")
    ablation_group_id = raw["ablation_group_id"]
    if ablation_group_id is not None:
        ablation_group_id = _nonempty(
            ablation_group_id, "manifest.ablation_group_id"
        )
    if comparison_kind == "tool_ablation" and ablation_group_id is None:
        raise ValueError("tool ablation requires ablation_group_id")
    candidate_config = raw["candidate_config"]
    if not isinstance(candidate_config, Mapping):
        raise ValueError("manifest.candidate_config must be an object")
    config = dict(candidate_config)
    model = _nonempty(
        config.get("exact_model_id"), "manifest.candidate_config.exact_model_id"
    )
    provider = _nonempty(
        config.get("provider_slug"), "manifest.candidate_config.provider_slug"
    )
    calibration = raw["calibration_gate"]
    if not isinstance(calibration, Mapping) or set(calibration) != CALIBRATION_GATE_KEYS:
        raise ValueError("manifest calibration gate keys are not canonical")
    if not isinstance(calibration["passed"], bool):
        raise ValueError("manifest calibration passed must be boolean")
    manifest = {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "variant_id": _nonempty(raw["variant_id"], "manifest.variant_id"),
        "model": model,
        "provider": provider,
        "comparison_kind": comparison_kind,
        "ablation_group_id": ablation_group_id,
        "standard_version": _nonempty(
            raw["standard_version"], "manifest.standard_version"
        ),
        **{
            key: _sha256_value(raw[key], f"manifest.{key}")
            for key in (
                "standard_sha256",
                "task_pack_identity_sha256",
                "pack_sha256",
                "tool_surface_sha256",
                "candidate_config_sha256",
                "prompt_policy_sha256",
                "verifier_policy_sha256",
                "result_matrix_sha256",
                "runtime_matrix_sha256",
                "result_binding_sha256",
                "business_observations_sha256",
            )
        },
        "verifier_policy_version": _nonempty(
            raw["verifier_policy_version"], "manifest.verifier_policy_version"
        ),
        "candidate_config": config,
        "calibration_gate": {
            "policy_version": _nonempty(
                calibration["policy_version"],
                "manifest.calibration_gate.policy_version",
            ),
            "evidence_sha256": _sha256_value(
                calibration["evidence_sha256"],
                "manifest.calibration_gate.evidence_sha256",
            ),
            "passed": calibration["passed"],
        },
        "attestation": dict(attestation),
    }
    if manifest["candidate_config_sha256"] != sha256(config):
        raise ValueError("candidate configuration hash differs from manifest object")
    if manifest["standard_version"] != public["standard_version"] or manifest[
        "standard_sha256"
    ] != public["standard_sha256"]:
        raise ValueError("run manifest differs from the public standard")
    if manifest["result_matrix_sha256"] != _matrix_sha256(result_rows):
        raise ValueError("result matrix hash differs from run manifest")
    if manifest["runtime_matrix_sha256"] != _matrix_sha256(run_rows):
        raise ValueError("runtime matrix hash differs from run manifest")
    if manifest["result_binding_sha256"] != _matrix_sha256(
        result_receipt_bindings
    ):
        raise ValueError("result binding hash differs from run manifest")
    if manifest["business_observations_sha256"] != sha256(
        sorted(business_observations, key=lambda row: str(row.get("metric_id")))
    ):
        raise ValueError("business observations hash differs from run manifest")
    return manifest


def summarize_company_candidate(
    candidate: Mapping[str, Any],
    *,
    manifest_signing_key: bytes,
    public_standard: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate one complete candidate and summarize all three evidence planes."""

    if candidate.get("schema_version") != COMPANY_INPUT_SCHEMA_VERSION:
        raise ValueError("unsupported company candidate schema_version")
    if set(candidate) != CANDIDATE_KEYS:
        raise ValueError("company candidate keys are not canonical")
    public = dict(public_standard or standard())
    result_rows = candidate.get("result_rows")
    run_rows = candidate.get("run_rows")
    result_receipt_bindings = candidate.get("result_receipt_bindings")
    business_rows = candidate.get("business_observations", [])
    if (
        not isinstance(result_rows, list)
        or not isinstance(run_rows, list)
        or not isinstance(result_receipt_bindings, list)
    ):
        raise ValueError("result rows, run rows, and result bindings must be lists")
    if not isinstance(business_rows, list):
        raise ValueError("business_observations must be a list")
    raw_manifest = candidate.get("manifest")
    if not isinstance(raw_manifest, Mapping):
        raise ValueError("manifest must be an object")
    manifest = _validate_run_manifest(
        raw_manifest,
        signing_key=manifest_signing_key,
        public=public,
        result_rows=result_rows,
        run_rows=run_rows,
        result_receipt_bindings=result_receipt_bindings,
        business_observations=business_rows,
    )
    model = manifest["model"]
    provider = manifest["provider"]
    quality = summarize_results(
        result_rows, model=model, provider=provider, public_standard=public
    )
    runs = _validate_run_rows(run_rows, public, manifest)
    _validate_result_bindings(
        result_receipt_bindings, public=public, runs=runs
    )
    result_by_cell = {
        (row["task_id"], row["trial"]): row for row in result_rows
    }
    for row in runs:
        result = result_by_cell[(row["task_id"], row["trial"])]
        if row["status"] != "completed" and (
            result["operational_success"] or result["exact_success"]
        ):
            raise ValueError(
                "non-completed runtime cannot be operational or exact success"
            )
        if (
            row["destructive_side_effect"]
            or row["false_completion"]
            or row["stuck_loop"]
        ) and (result["operational_success"] or result["exact_success"]):
            raise ValueError(
                "catastrophic runtime annotation contradicts scored success"
            )
        if (
            row["status"] in EXECUTION_INVALID_STATUSES
            and result["execution_valid"]
        ):
            raise ValueError(
                "policy or harness runtime status contradicts execution validity"
            )
        if row["status"] in COST_GATE_FAILED_STATUSES and row[
            "cost_cap_respected"
        ]:
            raise ValueError(
                "cost-gate-failed runtime cannot report a respected cost cap"
            )
        if not math.isclose(
            row["llm_inference_cost_usd"],
            float(result["llm_inference_cost_usd"]),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("runtime and result inference cost differ")
        if not math.isclose(
            row["paid_media_credits"],
            float(result["paid_media_credits"]),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("runtime and result paid-media credits differ")
        if row["cost_cap_respected"] is not result["cost_cap_respected"]:
            raise ValueError("runtime and result cost-cap status differ")
    if quality["score_status"] == "valid" and not manifest["calibration_gate"][
        "passed"
    ]:
        quality = {
            **quality,
            "score_status": "calibration_gate_failed",
            "official_score": None,
        }
    business = _business_observations(business_rows, manifest)

    operational_successes = quality["operational_success"]["trials"]
    totals = defaultdict(float)
    status_counts: defaultdict[str, int] = defaultdict(int)
    per_tool: dict[str, defaultdict[str, float]] = {}
    for row in runs:
        status_counts[row["status"]] += 1
        for key in (
            "input_tokens",
            "cached_input_tokens",
            "reasoning_tokens",
            "output_tokens",
            "inference_calls",
            "human_interventions",
        ):
            totals[key] += row[key]
        for key in (
            "false_completion",
            "destructive_side_effect",
            "context_limit_hit",
            "stuck_loop",
            "external_dependency_failure",
        ):
            totals[key] += int(row[key])
        for tool in row["tool_metrics"]:
            aggregate = per_tool.setdefault(tool["name"], defaultdict(float))
            for key, value in tool.items():
                if key != "name":
                    aggregate[key] += value

    tool_totals = defaultdict(float)
    for aggregate in per_tool.values():
        for key, value in aggregate.items():
            tool_totals[key] += value
    tools = []
    for name, aggregate in sorted(per_tool.items()):
        tools.append(
            {
                "name": name,
                "attempts": int(aggregate["attempts"]),
                "dispatched": int(aggregate["dispatched"]),
                "dispatch_success_rate": _rate(
                    aggregate["successes"], aggregate["dispatched"]
                ),
                "invalid_argument_rate": _rate(
                    aggregate["invalid_arguments"], aggregate["attempts"]
                ),
                "redundant_call_rate": _rate(
                    aggregate["redundant_calls"], aggregate["dispatched"]
                ),
                "same_tool_success_after_error_rate": _rate(
                    aggregate["same_tool_successes_after_error"],
                    aggregate["invalid_arguments"] + aggregate["errors"],
                ),
                "latency_seconds": round(aggregate["latency_seconds"], 6),
            }
        )

    cell_count = len(runs)
    total_cost = quality["cost"]["llm_inference_usd"]
    observed = set(business)
    return {
        "model": model,
        "provider": provider,
        "variant_id": manifest["variant_id"],
        "display_name": f"{model} [{manifest['variant_id']}]",
        "manifest": manifest,
        "quality": quality,
        "speed": {
            "end_to_end_seconds": {
                "mean": round(mean(row["latency_seconds"] for row in runs), 6),
                "p50": round(_percentile([row["latency_seconds"] for row in runs], 0.5), 6),
                "p95": round(_percentile([row["latency_seconds"] for row in runs], 0.95), 6),
            },
            "first_model_response_p50_seconds": round(
                _percentile(
                    [
                        row["first_model_response_seconds"]
                        for row in runs
                        if row["first_model_response_seconds"] is not None
                    ],
                    0.5,
                ),
                6,
            )
            if any(row["first_model_response_seconds"] is not None for row in runs)
            else None,
            "inference_seconds": round(sum(row["inference_seconds"] for row in runs), 6),
            "mcp_seconds": round(sum(row["mcp_seconds"] for row in runs), 6),
        },
        "economics": {
            "llm_inference_usd": total_cost,
            "mean_cost_per_trial_usd": round(total_cost / cell_count, 6),
            "cost_per_operational_success_usd": (
                None
                if operational_successes == 0
                else round(total_cost / operational_successes, 6)
            ),
            "tokens": {
                key: int(totals[key])
                for key in (
                    "input_tokens",
                    "cached_input_tokens",
                    "reasoning_tokens",
                    "output_tokens",
                )
            },
            "inference_calls": int(totals["inference_calls"]),
        },
        "tooling": {
            "attempts": int(tool_totals["attempts"]),
            "dispatched": int(tool_totals["dispatched"]),
            "dispatch_success_rate": _rate(
                tool_totals["successes"], tool_totals["dispatched"]
            ),
            "invalid_argument_rate": _rate(
                tool_totals["invalid_arguments"], tool_totals["attempts"]
            ),
            "redundant_call_rate": _rate(
                tool_totals["redundant_calls"], tool_totals["dispatched"]
            ),
            "same_tool_success_after_error_rate": _rate(
                tool_totals["same_tool_successes_after_error"],
                tool_totals["invalid_arguments"] + tool_totals["errors"],
            ),
            "by_tool": tools,
        },
        "reliability": {
            "status_counts": dict(sorted(status_counts.items())),
            "human_interventions": int(totals["human_interventions"]),
            "human_intervention_rate": _rate(
                sum(row["human_interventions"] > 0 for row in runs), cell_count
            ),
            **{
                f"{key}_rate": _rate(totals[key], cell_count)
                for key in (
                    "false_completion",
                    "destructive_side_effect",
                    "context_limit_hit",
                    "stuck_loop",
                    "external_dependency_failure",
                )
            },
        },
        "business": {
            "observations": business,
            "measured_metric_count": len(observed),
            "causal_metric_count": sum(
                row["claim_level"] == "causal" for row in business.values()
            ),
            "unmeasured_metric_ids": [
                metric["id"] for metric in BUSINESS_METRICS if metric["id"] not in observed
            ],
        },
    }


def build_company_report(
    candidates: Sequence[Mapping[str, Any]], *, manifest_signing_key: bytes
) -> dict[str, Any]:
    """Build a comparison report; deliberately do not choose one overall winner."""

    if not candidates:
        raise ValueError("at least one company candidate is required")
    summaries = [
        summarize_company_candidate(
            candidate, manifest_signing_key=manifest_signing_key
        )
        for candidate in candidates
    ]
    variant_ids = [row["manifest"]["variant_id"] for row in summaries]
    if len(variant_ids) != len(set(variant_ids)):
        raise ValueError("candidate variant IDs must be unique")
    shared_keys = (
        "standard_version",
        "standard_sha256",
        "prompt_policy_sha256",
        "verifier_policy_version",
        "verifier_policy_sha256",
        "task_pack_identity_sha256",
    )
    reference = summaries[0]["manifest"]
    for summary in summaries[1:]:
        manifest = summary["manifest"]
        if any(manifest[key] != reference[key] for key in shared_keys):
            raise ValueError("comparison candidates do not share evaluation policy")
        if manifest["calibration_gate"] != reference["calibration_gate"]:
            raise ValueError("comparison candidates do not share calibration evidence")
    standard_manifests = [
        summary["manifest"]
        for summary in summaries
        if summary["manifest"]["comparison_kind"] == "standard"
    ]
    if standard_manifests:
        standard_reference = standard_manifests[0]
        if any(
            manifest["pack_sha256"] != standard_reference["pack_sha256"]
            or manifest["tool_surface_sha256"]
            != standard_reference["tool_surface_sha256"]
            for manifest in standard_manifests[1:]
        ):
            raise ValueError("standard candidates do not share pack and tool surface")
    by_ablation_group: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for summary in summaries:
        group_id = summary["manifest"]["ablation_group_id"]
        if group_id is not None:
            by_ablation_group[group_id].append(summary["manifest"])
    for group_id, manifests in by_ablation_group.items():
        bases = [row for row in manifests if row["comparison_kind"] == "standard"]
        ablations = [
            row for row in manifests if row["comparison_kind"] == "tool_ablation"
        ]
        if len(bases) != 1 or not ablations:
            raise ValueError(
                f"ablation group {group_id} requires one standard base and an ablation"
            )
        base = bases[0]
        for ablation in ablations:
            if ablation["candidate_config_sha256"] != base["candidate_config_sha256"]:
                raise ValueError("tool ablation candidate configuration differs from base")
            if ablation["tool_surface_sha256"] == base["tool_surface_sha256"]:
                raise ValueError("tool ablation must change the tool surface")
    return {
        "schema_version": COMPANY_REPORT_SCHEMA_VERSION,
        "standard_version": summaries[0]["quality"]["standard_version"],
        "comparison_policy": {
            "overall_composite": None,
            "reason": "quality, operating economics, and business impact answer different decisions",
            "business_claim_rule": "observed and directional results are not causal",
        },
        "candidates": summaries,
        "business_metric_catalog": [dict(metric) for metric in BUSINESS_METRICS],
    }


def _format_rate(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _format_business_value(value: float, unit: str) -> str:
    if unit == "rate":
        return f"{value:.1%}"
    if unit in {"usd", "signed_usd"}:
        return f"${value:.2f}"
    return f"{value:.2f}"


def render_company_markdown(report: Mapping[str, Any]) -> str:
    """Render an executive-readable comparison with explicit denominators."""

    candidates = report["candidates"]
    lines = [
        "# AIVideo Agent Company Scorecard",
        "",
        "> Decision rule: compare quality, speed, cost, tool reliability, and business outcomes together. There is intentionally no opaque overall winner score.",
        "",
        "## Executive comparison",
        "",
        "| Candidate | Quality | Operational | Exact | Latency p50 / p95 | LLM cost | Cost / usable success | Tool success | Intervention |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in candidates:
        quality = row["quality"]
        speed = row["speed"]["end_to_end_seconds"]
        economics = row["economics"]
        official = quality["official_score"]
        score = "invalid" if official is None else f"{official:.2f}/100"
        cost_per_success = economics["cost_per_operational_success_usd"]
        lines.append(
            "| "
            + " | ".join(
                (
                    f"{row['display_name']} ({row['provider']})",
                    score,
                    _format_rate(quality["operational_success"]["rate"]),
                    _format_rate(quality["exact_success"]["rate"]),
                    f"{speed['p50']:.2f}s / {speed['p95']:.2f}s",
                    f"${economics['llm_inference_usd']:.4f}",
                    "n/a" if cost_per_success is None else f"${cost_per_success:.4f}",
                    _format_rate(row["tooling"]["dispatch_success_rate"]),
                    _format_rate(row["reliability"]["human_intervention_rate"]),
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## What success means",
            "",
            "- **Quality** is the existing 100-point benchmark: 100 tasks, one continuous point per task, three trials each.",
            "- **Operational success** requires a usable deliverable, critical-quality floors, satisfied constraints, and a valid execution.",
            "- **Exact success** requires every scored component to be perfect.",
            "- **Business impact is not included in the 100 points.** It needs production cohorts or experiments, not benchmark guesswork.",
            "",
            "## Speed and inference economics",
            "",
            "| Candidate | First response p50 | Inference time | MCP time | Inference calls | Input (cached) tokens | Reasoning tokens | Output tokens | Mean cost / trial |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in candidates:
        speed = row["speed"]
        economics = row["economics"]
        tokens = economics["tokens"]
        first_response = speed["first_model_response_p50_seconds"]
        lines.append(
            f"| {row['display_name']} | "
            f"{'n/a' if first_response is None else f'{first_response:.2f}s'} | "
            f"{speed['inference_seconds']:.2f}s | {speed['mcp_seconds']:.2f}s | "
            f"{economics['inference_calls']} | "
            f"{tokens['input_tokens']} ({tokens['cached_input_tokens']}) | "
            f"{tokens['reasoning_tokens']} | {tokens['output_tokens']} | "
            f"${economics['mean_cost_per_trial_usd']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Reliability incidents",
            "",
            "| Candidate | False completion | Destructive side effect | Context limit | Stuck loop | External dependency |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in candidates:
        reliability = row["reliability"]
        lines.append(
            "| "
            + " | ".join(
                [
                    row["display_name"],
                    *(
                        _format_rate(reliability[key])
                        for key in (
                            "false_completion_rate",
                            "destructive_side_effect_rate",
                            "context_limit_hit_rate",
                            "stuck_loop_rate",
                            "external_dependency_failure_rate",
                        )
                    ),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Tool performance", ""])
    for row in candidates:
        lines.extend(
            [
                f"### {row['display_name']}",
                "",
                "| Tool | Attempts | Dispatched | Success | Invalid arguments | Redundant | Same-tool success after error | Tool time |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        if not row["tooling"]["by_tool"]:
            lines.append("| No tool calls | 0 | 0 | n/a | n/a | n/a | n/a | 0.00s |")
        for tool in row["tooling"]["by_tool"]:
            lines.append(
                f"| {tool['name']} | {tool['attempts']} | {tool['dispatched']} | "
                f"{_format_rate(tool['dispatch_success_rate'])} | "
                f"{_format_rate(tool['invalid_argument_rate'])} | "
                f"{_format_rate(tool['redundant_call_rate'])} | "
                f"{_format_rate(tool['same_tool_success_after_error_rate'])} | "
                f"{tool['latency_seconds']:.2f}s |"
            )
        lines.append("")
    lines.extend(
        [
            "## Business outcome evidence",
            "",
            "Claim levels: **observed** describes a cohort, **directional** adds a comparison, and **causal** requires a valid experiment or equivalent design.",
            "",
            "| Business question | Direction | "
            + " | ".join(row["display_name"] for row in candidates)
            + " |",
            "|---|---:|" + "---:|" * len(candidates),
        ]
    )
    for metric in report["business_metric_catalog"]:
        cells = []
        for row in candidates:
            observation = row["business"]["observations"].get(metric["id"])
            if observation is None:
                cells.append("not measured")
            else:
                evidence = observation["evidence"]
                comparison = ""
                if observation["claim_level"] == "directional":
                    comparison = (
                        f", comparator={_format_business_value(evidence['comparator_value'], metric['unit'])}, "
                        f"delta={_format_business_value(evidence['absolute_delta'], metric['unit'])}"
                    )
                elif observation["claim_level"] == "causal":
                    comparison = (
                        f", control={_format_business_value(evidence['control_value'], metric['unit'])}, "
                        f"effect={_format_business_value(evidence['absolute_effect'], metric['unit'])}, "
                        f"CI=[{_format_business_value(evidence['effect_ci_lower'], metric['unit'])}, "
                        f"{_format_business_value(evidence['effect_ci_upper'], metric['unit'])}]"
                    )
                cells.append(
                    f"{_format_business_value(observation['value'], metric['unit'])} "
                    f"(n={observation['denominator']}, {observation['claim_level']}{comparison})"
                )
        lines.append(
            f"| {metric['label']} | {metric['direction']} | " + " | ".join(cells) + " |"
        )
    lines.extend(
        [
            "",
            "## Required interpretation",
            "",
            "- Compare model variants on the same sealed tasks, tool surface, trial count, and verifier version.",
            "- Treat external dependency failures separately from model failures, but keep both visible to product owners.",
            "- Never convert missing business data to zero. `not measured` means the production join does not exist yet.",
            "- Select a model from the quality, latency, cost, and reliability frontier for the target workflow; do not rank by an invented blended number.",
        ]
    )
    return "\n".join(lines) + "\n"
