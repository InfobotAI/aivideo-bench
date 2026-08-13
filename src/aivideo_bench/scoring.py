"""Additive, human-readable scoring for verified V3 task outcomes.

This contract intentionally does not replace historical task scores.  It exposes
the same component evidence without multiplying critical quality twice or using
constraints and cost policy as hidden quality multipliers.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from statistics import mean
from typing import Any


SCORING_CONTRACT_VERSION = "v3-interpretable-additive-v5"
SCORE_RELEASE_POLICY = (
    "after_exact_stage_terminal_receipts_interpretable_additive_v5"
)
QUALITY_WEIGHTS = {
    "critical": 0.60,
    "criteria": 0.30,
    "robustness": 0.10,
}
OPERATIONAL_QUALITY_MINIMUM = 0.75
OPERATIONAL_CRITICAL_MINIMUM = 0.80
COST_CEILING_RESOLUTION = "candidate_cost_ceiling_exceeded_fail_closed_zero"


def _unit(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric, not boolean")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be numeric") from None
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise ValueError(f"{label} must be in [0,1]")
    return number


def _group_values(evidence: Mapping[str, Any], group: str) -> list[float]:
    raw = evidence.get(group) or {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"{group} must be an object")
    values = [_unit(value, f"{group}.{key}") for key, value in sorted(raw.items())]
    if not values:
        raise ValueError(f"at least one {group} check is required")
    return values


def _catastrophic_gate(evidence: Mapping[str, Any]) -> float:
    raw = evidence.get("catastrophic") or {}
    if not isinstance(raw, Mapping):
        raise ValueError("catastrophic must be an object")
    values = [
        _unit(value, f"catastrophic.{key}") for key, value in sorted(raw.items())
    ]
    if any(value not in {0.0, 1.0} for value in values):
        raise ValueError("catastrophic checks must be binary")
    return min(values, default=1.0)


def classify_execution(cell: Mapping[str, Any]) -> dict[str, Any]:
    """Separate hard execution invalidity from a cost-cap overrun."""

    resolution = str(cell.get("score_resolution_policy") or "")
    policy_violations = int(cell.get("policy_violations", 0))
    harness_incidents = int(cell.get("harness_incidents", 0))
    paid_media_credits = float(cell.get("paid_media_credits", 0.0))
    cost_cap_respected = resolution != COST_CEILING_RESOLUTION
    reasons: list[str] = []
    if harness_incidents:
        reasons.append("harness_incident")
    if paid_media_credits > 0:
        reasons.append("paid_media_credit_use")
    if policy_violations and cost_cap_respected:
        reasons.append("non_cost_policy_violation")
    return {
        "execution_valid": not reasons,
        "execution_invalidity_reasons": reasons,
        "cost_cap_respected": cost_cap_respected,
        "cost_overrun": not cost_cap_respected,
        "score_resolution_policy": resolution,
    }


def score_interpretable_task(
    evidence: Mapping[str, Any], *, cell: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Return additive quality plus explicit operational and exact success."""

    deliverable = _unit(evidence.get("deliverable"), "deliverable")
    values = {
        group: _group_values(evidence, group)
        for group in ("critical", "criteria", "constraints", "robustness")
    }
    group_scores = {group: mean(group_values) for group, group_values in values.items()}
    contributions = {
        group: QUALITY_WEIGHTS[group] * group_scores[group]
        for group in QUALITY_WEIGHTS
    }
    quality = sum(contributions.values())
    catastrophic_gate = _catastrophic_gate(evidence)
    execution = classify_execution(cell or {})
    critical_floor = min(values["critical"])
    operational_success = (
        execution["execution_valid"]
        and catastrophic_gate == 1.0
        and deliverable == 1.0
        and quality >= OPERATIONAL_QUALITY_MINIMUM
        and group_scores["critical"] >= OPERATIONAL_CRITICAL_MINIMUM
        and critical_floor > 0.0
        and group_scores["constraints"] == 1.0
    )
    exact_success = (
        execution["execution_valid"]
        and catastrophic_gate == 1.0
        and deliverable == 1.0
        and all(
            component == 1.0
            for group_values in values.values()
            for component in group_values
        )
    )
    if not execution["execution_valid"]:
        success_status = "invalid_execution"
    elif exact_success:
        success_status = "exact_success"
    elif operational_success:
        success_status = "operational_success"
    else:
        success_status = "partial"
    return {
        "scoring_contract_version": SCORING_CONTRACT_VERSION,
        "quality_weights": dict(QUALITY_WEIGHTS),
        "quality_score": round(quality, 9),
        "group_scores": {
            group: round(score, 9) for group, score in group_scores.items()
        },
        "weighted_contributions": {
            group: round(score, 9) for group, score in contributions.items()
        },
        "deliverable_score": deliverable,
        "critical_component_floor": round(critical_floor, 9),
        "catastrophic_gate": catastrophic_gate,
        "operational_success": operational_success,
        "exact_success": exact_success,
        "success_status": success_status,
        **execution,
    }
