from aivideo_bench.scoring import (
    COST_CEILING_RESOLUTION,
    classify_execution,
    score_interpretable_task,
)


def _evidence(**overrides) -> dict:
    evidence = {
        "deliverable": 1.0,
        "critical": {"a": 1.0, "b": 1.0},
        "criteria": {"a": 1.0},
        "constraints": {"a": 1.0},
        "robustness": {"a": 1.0},
    }
    evidence.update(overrides)
    return evidence


def test_additive_quality_has_explicit_success_levels():
    exact = score_interpretable_task(_evidence())
    partial = score_interpretable_task(
        _evidence(
            critical={"a": 1.0, "b": 0.5},
            criteria={"a": 0.5},
            robustness={"a": 0.5},
        )
    )

    assert exact["quality_score"] == 1.0
    assert exact["success_status"] == "exact_success"
    assert exact["operational_success"] is True
    assert exact["exact_success"] is True
    assert partial["quality_score"] == 0.65
    assert partial["success_status"] == "partial"
    assert partial["operational_success"] is False


def test_cost_overrun_is_separate_from_verified_quality():
    cell = {
        "score_resolution_policy": COST_CEILING_RESOLUTION,
        "policy_violations": 1,
        "harness_incidents": 0,
        "paid_media_credits": 0.0,
    }

    result = score_interpretable_task(_evidence(), cell=cell)

    assert result["quality_score"] == 1.0
    assert result["execution_valid"] is True
    assert result["cost_overrun"] is True
    assert result["exact_success"] is True


def test_non_cost_policy_violation_invalidates_success_not_diagnostic_quality():
    cell = {
        "score_resolution_policy": "blocked_task_scope_attempt_fail_closed_zero",
        "policy_violations": 1,
    }

    result = score_interpretable_task(_evidence(), cell=cell)

    assert result["quality_score"] == 1.0
    assert result["success_status"] == "invalid_execution"
    assert result["execution_invalidity_reasons"] == ["non_cost_policy_violation"]
    assert result["operational_success"] is False
    assert result["exact_success"] is False


def test_operational_success_requires_no_zero_critical_component():
    result = score_interpretable_task(
        _evidence(
            critical={"a": 1.0, "b": 1.0, "c": 1.0, "d": 1.0, "e": 0.0},
            criteria={"a": 1.0},
            robustness={"a": 1.0},
        )
    )

    assert result["quality_score"] == 0.88
    assert result["group_scores"]["critical"] == 0.8
    assert result["operational_success"] is False


def test_execution_classifier_rejects_harness_and_paid_media():
    result = classify_execution(
        {
            "harness_incidents": 1,
            "paid_media_credits": 0.25,
            "policy_violations": 0,
        }
    )

    assert result["execution_valid"] is False
    assert result["execution_invalidity_reasons"] == [
        "harness_incident",
        "paid_media_credit_use",
    ]
