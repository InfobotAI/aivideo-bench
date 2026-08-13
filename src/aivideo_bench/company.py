"""Company-facing model comparison without inventing a composite business score."""

from __future__ import annotations

import math
from collections import defaultdict
from statistics import mean
from typing import Any, Mapping, Sequence

from .results import summarize_results
from .standard import standard


COMPANY_INPUT_SCHEMA_VERSION = "aivideo-bench-company-candidate-v1"
COMPANY_REPORT_SCHEMA_VERSION = "aivideo-bench-company-report-v1"
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
        "recovered_errors",
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
    }
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
            "recovered_errors",
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
    recoverable = counts["invalid_arguments"] + counts["errors"]
    if counts["recovered_errors"] > recoverable:
        raise ValueError(f"{label}: recovered errors exceed observed errors")
    return {
        "name": name,
        **counts,
        "latency_seconds": _number(raw["latency_seconds"], f"{label}.latency_seconds"),
    }


def _validate_run_rows(
    rows: Sequence[Mapping[str, Any]], public: Mapping[str, Any]
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
        if not status:
            raise ValueError(f"{label}.status must be non-empty")
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
                    key: _number(raw[key], f"{label}.{key}")
                    for key in (
                        "latency_seconds",
                        "inference_seconds",
                        "mcp_seconds",
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
                **booleans,
            }
        )
        row = validated[-1]
        if (
            row["first_model_response_seconds"] is not None
            and row["first_model_response_seconds"] > row["latency_seconds"]
        ):
            raise ValueError(f"{label}: first model response exceeds total latency")
        if row["cached_input_tokens"] > row["input_tokens"]:
            raise ValueError(f"{label}: cached input tokens exceed input tokens")
    if seen != expected:
        raise ValueError(
            "run telemetry must contain exactly 300 canonical cells "
            f"(missing={len(expected - seen)}, extra={len(seen - expected)})"
        )
    return validated


def run_row_from_receipt(
    receipt: Mapping[str, Any],
    *,
    trial: int,
    annotations: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert a v2 execution receipt plus verifier annotations to telemetry."""

    if receipt.get("schema_version") != "aivideo-bench-candidate-execution-v2":
        raise ValueError("company telemetry requires a v2 candidate execution receipt")
    notes = dict(annotations or {})
    allowed_notes = {
        "human_interventions",
        "false_completion",
        "destructive_side_effect",
        "context_limit_hit",
        "stuck_loop",
        "external_dependency_failure",
        "redundant_calls_by_tool",
    }
    if set(notes) - allowed_notes:
        raise ValueError("unknown verifier annotation keys")
    redundant = notes.get("redundant_calls_by_tool", {})
    if not isinstance(redundant, Mapping):
        raise ValueError("redundant_calls_by_tool must be an object")
    token_usage = receipt.get("token_usage")
    receipt_tools = receipt.get("tool_metrics")
    if not isinstance(token_usage, Mapping) or not isinstance(receipt_tools, list):
        raise ValueError("v2 receipt is missing token or tool telemetry")
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
                "recovered_errors": tool.get("recovered_errors"),
                "latency_seconds": tool.get("latency_seconds"),
            }
        )
    row = {
        "task_id": receipt.get("task_id"),
        "trial": trial,
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
        "tool_metrics": tools,
        "human_interventions": notes.get("human_interventions", 0),
        "false_completion": notes.get("false_completion", False),
        "destructive_side_effect": notes.get("destructive_side_effect", False),
        "context_limit_hit": notes.get("context_limit_hit", False),
        "stuck_loop": notes.get("stuck_loop", False),
        "external_dependency_failure": notes.get(
            "external_dependency_failure", False
        ),
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
        if claim_level == "causal" and source != "experiment":
            raise ValueError(f"{label}: causal claims require experiment source")
        if not window:
            raise ValueError(f"{label}.window must be non-empty")
        expected_value = numerator / denominator
        if metric["unit"] == "per_100":
            expected_value *= 100
        if not math.isclose(value, expected_value, rel_tol=1e-6, abs_tol=1e-9):
            raise ValueError(f"{label}: value does not match numerator and denominator")
        observations[metric_id] = {
            "value": value,
            "numerator": numerator,
            "denominator": denominator,
            "window": window,
            "claim_level": claim_level,
            "source": source,
        }
    return observations


def summarize_company_candidate(
    candidate: Mapping[str, Any], *, public_standard: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Validate one complete candidate and summarize all three evidence planes."""

    if candidate.get("schema_version") != COMPANY_INPUT_SCHEMA_VERSION:
        raise ValueError("unsupported company candidate schema_version")
    public = dict(public_standard or standard())
    model = str(candidate.get("model") or "").strip()
    provider = str(candidate.get("provider") or "").strip()
    if not model or not provider:
        raise ValueError("model and provider must be non-empty")
    result_rows = candidate.get("result_rows")
    run_rows = candidate.get("run_rows")
    business_rows = candidate.get("business_observations", [])
    if not isinstance(result_rows, list) or not isinstance(run_rows, list):
        raise ValueError("result_rows and run_rows must be lists")
    if not isinstance(business_rows, list):
        raise ValueError("business_observations must be a list")
    quality = summarize_results(
        result_rows, model=model, provider=provider, public_standard=public
    )
    runs = _validate_run_rows(run_rows, public)
    business = _business_observations(business_rows)

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
                "error_recovery_rate": _rate(
                    aggregate["recovered_errors"],
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
            "error_recovery_rate": _rate(
                tool_totals["recovered_errors"],
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


def build_company_report(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Build a comparison report; deliberately do not choose one overall winner."""

    if not candidates:
        raise ValueError("at least one company candidate is required")
    summaries = [summarize_company_candidate(candidate) for candidate in candidates]
    identities = [(row["model"], row["provider"]) for row in summaries]
    if len(identities) != len(set(identities)):
        raise ValueError("candidate model/provider identities must be unique")
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
                    f"{row['model']} ({row['provider']})",
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
            f"| {row['model']} | "
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
                    row["model"],
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
                f"### {row['model']}",
                "",
                "| Tool | Attempts | Dispatched | Success | Invalid arguments | Redundant | Recovery | Tool time |",
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
                f"{_format_rate(tool['error_recovery_rate'])} | "
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
            + " | ".join(row["model"] for row in candidates)
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
                cells.append(
                    f"{_format_business_value(observation['value'], metric['unit'])} "
                    f"(n={observation['denominator']}, {observation['claim_level']})"
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
