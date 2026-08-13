import copy

from aivideo_bench.standard import (
    DOMAIN_CAPABILITIES,
    PRIVATE_KEYS,
    STANDARD_POLICY,
    coverage_summary,
    standard,
    validate_standard,
)


def test_standard_is_exactly_100_equal_continuous_points():
    data = standard()

    assert validate_standard(data) == []
    assert coverage_summary(data) == {
        "task_count": 100,
        "points": 100.0,
        "domains": {domain: 10 for domain in sorted(DOMAIN_CAPABILITIES)},
        "tiers": {
            "adversarial": 30,
            "foundation": 20,
            "frontier": 20,
            "integration": 30,
        },
    }
    assert all(task["points"] == 1.0 for task in data["tasks"])


def test_public_registry_contains_no_private_task_material():
    data = standard()

    serialized_keys = set()
    stack = [data]
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            serialized_keys.update(value)
            stack.extend(value.values())
        elif isinstance(value, list):
            stack.extend(value)

    assert serialized_keys.isdisjoint(PRIVATE_KEYS)
    assert data["official_eligible"] is False


def test_calibration_policy_encodes_a_difficult_but_human_solvable_bench():
    calibration = STANDARD_POLICY["calibration"]

    assert calibration["expert_panel_minimum"] == 90.0
    assert calibration["frontier_model_panel_median_maximum"] == 50.0
    assert calibration["no_op_maximum"] < calibration["render_only_maximum"]
    assert calibration["render_only_maximum"] < calibration["scripted_partial_maximum"]


def test_standard_returns_a_defensive_copy():
    first = standard()
    first["tasks"][0]["points"] = 0.0

    assert standard()["tasks"][0]["points"] == 1.0


def test_validation_detects_reordering_and_private_leakage():
    altered = copy.deepcopy(standard())
    altered["tasks"][0], altered["tasks"][1] = altered["tasks"][1], altered["tasks"][0]
    altered["tasks"][0]["prompt"] = "secret"

    errors = validate_standard(altered)

    assert "standard hash mismatch" in errors
    assert any("ordinal mismatch" in error for error in errors)
    assert "public standard leaks private keys: ['prompt']" in errors
