import copy

import pytest

from aivideo_bench.private_pack import (
    seal_private_pack,
    validate_private_pack,
)
from aivideo_bench.results import render_markdown, summarize_results
from aivideo_bench.standard import standard


KEY = b"k" * 32
COMMITMENT = "a" * 64


def _tasks(*, tool_name: str = "place_clips") -> list[dict]:
    return [
        {
            "task_id": task["id"],
            "blueprint_sha256": task["blueprint_sha256"],
            "prompt": f"Private instructions for {task['id']}",
            "allowed_tools": sorted(["get_context", tool_name]),
            "fixture_sha256": COMMITMENT,
            "reference_sha256": "b" * 64,
            "verifier_sha256": "c" * 64,
        }
        for task in standard()["tasks"]
    ]


def _pack(*, tool_name: str = "place_clips") -> dict:
    return seal_private_pack(
        _tasks(tool_name=tool_name),
        tool_names=["get_context", tool_name],
        zero_media_cost_tools=["get_context", tool_name],
        tool_surface_sha256="d" * 64,
        key_id="test-owner-key",
        signing_key=KEY,
    )


def _rows(*, invalid_cell: tuple[str, int] | None = None, over_cap: bool = False):
    rows = []
    for task in standard()["tasks"]:
        for trial in range(1, 4):
            cell = (task["id"], trial)
            invalid = cell == invalid_cell
            rows.append(
                {
                    "task_id": task["id"],
                    "trial": trial,
                    "quality_score": 0.5,
                    "operational_success": not invalid and trial == 1,
                    "exact_success": False,
                    "execution_valid": not invalid,
                    "cost_cap_respected": not over_cap,
                    "llm_inference_cost_usd": 0.01,
                    "paid_media_credits": 1.0 if invalid else 0.0,
                }
            )
    return rows


def test_private_pack_binds_all_tasks_and_real_tool_surface():
    pack = _pack()

    assert validate_private_pack(pack, signing_key=KEY) == []
    assert len(pack["tasks"]) == 100
    assert pack["tool_surface"]["tool_names"] == ["get_context", "place_clips"]
    assert pack["tool_surface"]["zero_media_cost_tools"] == [
        "get_context",
        "place_clips",
    ]
    assert "signing_key" not in repr(pack)


def test_private_pack_detects_post_seal_mutation():
    pack = _pack()
    pack["tasks"][0]["prompt"] = "tampered"

    errors = validate_private_pack(pack, signing_key=KEY)

    assert "private pack hash mismatch" in errors
    assert "private pack HMAC mismatch" in errors


def test_private_pack_requires_canonical_task_and_tool_order():
    pack = _pack()
    pack["tasks"][0], pack["tasks"][1] = pack["tasks"][1], pack["tasks"][0]
    pack["tasks"][2]["allowed_tools"].reverse()

    errors = validate_private_pack(pack, signing_key=KEY)

    assert "private tasks must follow public registry order" in errors
    assert any("allowed_tools must be sorted canonically" in error for error in errors)


def test_synthetic_caption_tool_is_rejected():
    with pytest.raises(ValueError, match="synthetic tools are prohibited"):
        _pack(tool_name="add_captions")


def test_unclassified_media_tool_is_rejected_before_inference():
    with pytest.raises(ValueError, match="not classified zero-media-cost"):
        seal_private_pack(
            _tasks(),
            tool_names=["get_context", "place_clips"],
            zero_media_cost_tools=["get_context"],
            tool_surface_sha256="d" * 64,
            key_id="test-owner-key",
            signing_key=KEY,
        )


def test_complete_result_is_a_50_point_score_with_separate_success():
    summary = summarize_results(_rows(), model="Luna", provider="Codex")

    assert summary["official_score"] == 50.0
    assert summary["diagnostic_quality_score"] == 50.0
    assert summary["operational_success"] == {"trials": 100, "rate": 0.333333}
    assert summary["exact_success"] == {"trials": 0, "rate": 0.0}
    assert summary["cost"]["llm_inference_usd"] == 3.0
    assert summary["cost"]["paid_media_credits"] == 0.0
    assert all(score == 5.0 for score in summary["domain_scores"].values())


def test_invalid_execution_removes_official_score_but_keeps_diagnostic_quality():
    first_task = standard()["tasks"][0]["id"]
    summary = summarize_results(
        _rows(invalid_cell=(first_task, 1)), model="Luna", provider="Codex"
    )

    assert summary["score_status"] == "invalid_execution"
    assert summary["official_score"] is None
    assert summary["diagnostic_quality_score"] == 50.0
    assert summary["invalid_execution"]["trials"] == 1
    assert summary["cost"]["paid_media_credits"] == 1.0


def test_cost_overrun_is_visible_without_erasing_verified_quality():
    summary = summarize_results(
        _rows(over_cap=True), model="Luna", provider="Codex"
    )

    assert summary["official_score"] is None
    assert summary["diagnostic_quality_score"] == 50.0
    assert summary["score_status"] == "cost_gate_failed"
    assert summary["cost"]["cap_respected"] is False
    assert summary["cost"]["over_cap_trials"] == 300


def test_human_readable_report_explains_score_success_and_cost():
    report = render_markdown(
        summarize_results(_rows(), model="Luna", provider="Codex")
    )

    assert "Official score: **50.00/100**" in report
    assert "Operational success: **100/300 trials** (33.3%)" in report
    assert "Exact success: **0/300 trials** (0.0%)" in report
    assert "LLM inference cost: **$3.0000**" in report
    assert "Paid media credits: **0.0000**" in report


def test_result_matrix_must_be_complete_and_unique():
    with pytest.raises(ValueError, match="exactly 300 canonical cells"):
        summarize_results(_rows()[:-1], model="Luna", provider="Codex")

    rows = _rows()
    rows[-1] = copy.deepcopy(rows[0])
    with pytest.raises(ValueError, match="duplicate result cell"):
        summarize_results(rows, model="Luna", provider="Codex")
