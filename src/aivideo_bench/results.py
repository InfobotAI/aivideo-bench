"""Human-readable aggregation for complete AIVideo Bench result matrices."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from statistics import mean
from typing import Any, Mapping, Sequence

from .standard import standard


RESULT_ROW_KEYS = frozenset(
    {
        "task_id",
        "trial",
        "quality_score",
        "operational_success",
        "exact_success",
        "execution_valid",
        "cost_cap_respected",
        "llm_inference_cost_usd",
        "paid_media_credits",
    }
)
RESULT_REPORT_CONTRACT_VERSION = "aivideo-bench-result-report-v2-cost-gated"


def _number(value: Any, *, label: str, minimum: float, maximum: float | None = None) -> float:
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


def summarize_results(
    rows: Sequence[Mapping[str, Any]],
    *,
    model: str,
    provider: str,
    public_standard: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate 300 result rows and expose score, success, validity, and cost."""

    if not model.strip() or not provider.strip():
        raise ValueError("model and provider must be non-empty")
    public = dict(public_standard or standard())
    public_tasks = {task["id"]: task for task in public["tasks"]}
    expected_cells = {
        (task_id, trial)
        for task_id in public_tasks
        for trial in range(1, int(public["policy"]["trials_per_task"]) + 1)
    }
    seen: set[tuple[str, int]] = set()
    by_task: dict[str, list[float]] = defaultdict(list)
    operational_count = 0
    exact_count = 0
    invalid_count = 0
    cost_overrun_count = 0
    llm_cost = 0.0
    paid_media = 0.0
    invalid_reasons: Counter[str] = Counter()
    for index, raw in enumerate(rows, 1):
        if set(raw) != RESULT_ROW_KEYS:
            raise ValueError(f"row-{index}: result row keys are not canonical")
        task_id = str(raw["task_id"])
        if task_id not in public_tasks:
            raise ValueError(f"row-{index}: unknown task ID {task_id}")
        trial = raw["trial"]
        if isinstance(trial, bool) or not isinstance(trial, int) or trial not in {1, 2, 3}:
            raise ValueError(f"row-{index}: trial must be 1, 2, or 3")
        cell = (task_id, trial)
        if cell in seen:
            raise ValueError(f"row-{index}: duplicate result cell {cell}")
        seen.add(cell)
        quality = _number(
            raw["quality_score"],
            label=f"row-{index}.quality_score",
            minimum=0,
            maximum=1,
        )
        row_cost = _number(
            raw["llm_inference_cost_usd"],
            label=f"row-{index}.llm_inference_cost_usd",
            minimum=0,
        )
        row_media = _number(
            raw["paid_media_credits"],
            label=f"row-{index}.paid_media_credits",
            minimum=0,
        )
        booleans = {
            key: raw[key]
            for key in (
                "operational_success",
                "exact_success",
                "execution_valid",
                "cost_cap_respected",
            )
        }
        if any(not isinstance(value, bool) for value in booleans.values()):
            raise ValueError(f"row-{index}: status fields must be booleans")
        if booleans["exact_success"] and not booleans["operational_success"]:
            raise ValueError(f"row-{index}: exact success requires operational success")
        if not booleans["execution_valid"] and (
            booleans["operational_success"] or booleans["exact_success"]
        ):
            raise ValueError(f"row-{index}: invalid execution cannot be successful")
        if row_media > 0 and booleans["execution_valid"]:
            raise ValueError(f"row-{index}: paid media use must invalidate execution")
        by_task[task_id].append(quality)
        operational_count += int(booleans["operational_success"])
        exact_count += int(booleans["exact_success"])
        invalid_count += int(not booleans["execution_valid"])
        cost_overrun_count += int(not booleans["cost_cap_respected"])
        llm_cost += row_cost
        paid_media += row_media
        if not booleans["execution_valid"]:
            invalid_reasons["execution_invalid"] += 1
        if row_media > 0:
            invalid_reasons["paid_media_use"] += 1
    missing = expected_cells - seen
    extra = seen - expected_cells
    if missing or extra:
        raise ValueError(
            f"result matrix must contain exactly 300 canonical cells "
            f"(missing={len(missing)}, extra={len(extra)})"
        )
    task_scores = {task_id: mean(values) for task_id, values in by_task.items()}
    diagnostic_score = sum(task_scores.values())
    domain_scores: dict[str, float] = {}
    for domain in (row["id"] for row in public["domains"]):
        domain_scores[domain] = sum(
            task_scores[task_id]
            for task_id, task in public_tasks.items()
            if task["domain"] == domain
        )
    score_valid = (
        invalid_count == 0
        and paid_media == 0
        and cost_overrun_count == 0
    )
    if invalid_count or paid_media:
        score_status = "invalid_execution"
    elif cost_overrun_count:
        score_status = "cost_gate_failed"
    else:
        score_status = "valid"
    cell_count = len(expected_cells)
    return {
        "model": model,
        "provider": provider,
        "result_report_contract_version": RESULT_REPORT_CONTRACT_VERSION,
        "standard_version": public["standard_version"],
        "standard_sha256": public["standard_sha256"],
        "score_status": score_status,
        "official_score": round(diagnostic_score, 6) if score_valid else None,
        "diagnostic_quality_score": round(diagnostic_score, 6),
        "score_maximum": 100.0,
        "task_count": len(task_scores),
        "trial_count": cell_count,
        "operational_success": {
            "trials": operational_count,
            "rate": round(operational_count / cell_count, 6),
        },
        "exact_success": {
            "trials": exact_count,
            "rate": round(exact_count / cell_count, 6),
        },
        "invalid_execution": {
            "trials": invalid_count,
            "reasons": dict(sorted(invalid_reasons.items())),
        },
        "cost": {
            "llm_inference_usd": round(llm_cost, 6),
            "paid_media_credits": round(paid_media, 6),
            "cap_respected": cost_overrun_count == 0,
            "over_cap_trials": cost_overrun_count,
        },
        "domain_scores": {
            domain: round(score, 6) for domain, score in domain_scores.items()
        },
        "task_scores": {
            task_id: round(score, 9) for task_id, score in task_scores.items()
        },
    }


def render_markdown(summary: Mapping[str, Any]) -> str:
    """Render the result in language intended for humans, not score auditors."""

    valid = summary["score_status"] == "valid"
    displayed_score = (
        summary["official_score"] if valid else summary["diagnostic_quality_score"]
    )
    score_label = "Official score" if valid else "Diagnostic quality (invalid run)"
    operational = summary["operational_success"]
    exact = summary["exact_success"]
    invalid = summary["invalid_execution"]
    cost = summary["cost"]
    lines = [
        f"# {summary['model']} — AIVideo Bench",
        "",
        f"- {score_label}: **{displayed_score:.2f}/100**",
        f"- Operational success: **{operational['trials']}/{summary['trial_count']} trials** "
        f"({operational['rate']:.1%})",
        f"- Exact success: **{exact['trials']}/{summary['trial_count']} trials** "
        f"({exact['rate']:.1%})",
        f"- Invalid executions: **{invalid['trials']}**",
        f"- LLM inference cost: **${cost['llm_inference_usd']:.4f}**",
        f"- Paid media credits: **{cost['paid_media_credits']:.4f}**",
        f"- Cost cap respected: **{'yes' if cost['cap_respected'] else 'no'}**",
        "",
        "## Domain scores",
        "",
    ]
    for domain, score in summary["domain_scores"].items():
        lines.append(f"- {domain.replace('_', ' ').title()}: {score:.2f}/10")
    lines.extend(
        [
            "",
            "Quality is continuous partial credit. Operational and exact success are "
            "reported separately; they are not inferred from the total score.",
        ]
    )
    return "\n".join(lines) + "\n"
