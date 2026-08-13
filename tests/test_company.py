import copy
import json

import pytest

from aivideo_bench.company import (
    BUSINESS_METRICS,
    COMPANY_INPUT_SCHEMA_VERSION,
    build_company_report,
    metric_catalog,
    render_company_markdown,
    run_row_from_receipt,
    summarize_company_candidate,
)
from aivideo_bench.cli import main
from aivideo_bench.standard import standard


def _result_rows(*, quality=0.8, operational=True, cost=0.01):
    return [
        {
            "task_id": task["id"],
            "trial": trial,
            "quality_score": quality,
            "operational_success": operational,
            "exact_success": quality == 1.0,
            "execution_valid": True,
            "cost_cap_respected": True,
            "llm_inference_cost_usd": cost,
            "paid_media_credits": 0.0,
        }
        for task in standard()["tasks"]
        for trial in range(1, 4)
    ]


def _run_rows():
    rows = []
    for task_index, task in enumerate(standard()["tasks"]):
        for trial in range(1, 4):
            first = task_index == 0 and trial == 1
            rows.append(
                {
                    "task_id": task["id"],
                    "trial": trial,
                    "status": "completed",
                    "latency_seconds": 2.0,
                    "first_model_response_seconds": 0.5,
                    "inference_seconds": 1.25,
                    "mcp_seconds": 0.5,
                    "input_tokens": 100,
                    "cached_input_tokens": 20,
                    "reasoning_tokens": 10,
                    "output_tokens": 25,
                    "inference_calls": 2,
                    "tool_metrics": [
                        {
                            "name": "place_clips",
                            "attempts": 3 if first else 2,
                            "dispatched": 2,
                            "successes": 1,
                            "errors": 1,
                            "invalid_arguments": 1 if first else 0,
                            "redundant_calls": 0,
                            "recovered_errors": 2 if first else 1,
                            "latency_seconds": 0.5,
                        }
                    ],
                    "human_interventions": 1 if first else 0,
                    "false_completion": first,
                    "destructive_side_effect": False,
                    "context_limit_hit": False,
                    "stuck_loop": False,
                    "external_dependency_failure": first,
                }
            )
    return rows


def _business_observation(metric_id="task_to_usable_project_rate"):
    return {
        "metric_id": metric_id,
        "value": 0.72,
        "numerator": 72,
        "denominator": 100,
        "window": "2026-07-01/2026-07-31",
        "claim_level": "observed",
        "source": "production_telemetry",
    }


def _candidate(model="Candidate A", *, business=True):
    return {
        "schema_version": COMPANY_INPUT_SCHEMA_VERSION,
        "model": model,
        "provider": "Provider A",
        "result_rows": _result_rows(),
        "run_rows": _run_rows(),
        "business_observations": [_business_observation()] if business else [],
    }


def test_company_summary_keeps_quality_speed_cost_tools_and_business_separate():
    summary = summarize_company_candidate(_candidate())

    assert summary["quality"]["official_score"] == 80.0
    assert summary["speed"]["end_to_end_seconds"] == {
        "mean": 2.0,
        "p50": 2.0,
        "p95": 2.0,
    }
    assert summary["economics"]["llm_inference_usd"] == 3.0
    assert summary["economics"]["cost_per_operational_success_usd"] == 0.01
    assert summary["economics"]["tokens"]["reasoning_tokens"] == 3000
    assert summary["tooling"]["attempts"] == 601
    assert summary["tooling"]["dispatch_success_rate"] == 0.5
    assert summary["tooling"]["error_recovery_rate"] == 1.0
    assert summary["reliability"]["false_completion_rate"] == 0.003333
    assert summary["reliability"]["external_dependency_failure_rate"] == 0.003333
    assert summary["business"]["measured_metric_count"] == 1
    assert summary["business"]["causal_metric_count"] == 0


def test_report_refuses_an_opaque_overall_composite_and_marks_missing_data():
    report = build_company_report(
        [_candidate("Candidate A"), _candidate("Candidate B", business=False)]
    )
    rendered = render_company_markdown(report)

    assert report["comparison_policy"]["overall_composite"] is None
    assert "There is intentionally no opaque overall winner score" in rendered
    assert "Business impact is not included in the 100 points" in rendered
    assert "72.0% (n=100, observed)" in rendered
    assert "not measured" in rendered
    assert "External dependency" in rendered
    assert "Speed and inference economics" in rendered
    assert "30000 (6000)" in rendered


def test_runtime_matrix_must_be_complete_and_unique():
    candidate = _candidate()
    candidate["run_rows"] = candidate["run_rows"][:-1]

    with pytest.raises(ValueError, match="exactly 300 canonical cells"):
        summarize_company_candidate(candidate)

    candidate = _candidate()
    candidate["run_rows"][-1] = copy.deepcopy(candidate["run_rows"][0])
    with pytest.raises(ValueError, match="duplicate result cell"):
        summarize_company_candidate(candidate)


def test_tool_accounting_rejects_inconsistent_or_noncanonical_data():
    candidate = _candidate()
    candidate["run_rows"][0]["tool_metrics"][0]["successes"] = 2

    with pytest.raises(ValueError, match="successes plus errors must equal dispatched"):
        summarize_company_candidate(candidate)

    candidate = _candidate()
    candidate["run_rows"][0]["tool_metrics"][0]["surprise"] = 1
    with pytest.raises(ValueError, match="keys are not canonical"):
        summarize_company_candidate(candidate)


def test_business_observations_require_registered_metrics_and_honest_claims():
    candidate = _candidate()
    candidate["business_observations"][0]["metric_id"] = "vibes"
    with pytest.raises(ValueError, match="unknown metric ID"):
        summarize_company_candidate(candidate)

    candidate = _candidate()
    candidate["business_observations"][0]["claim_level"] = "definitely_caused"
    with pytest.raises(ValueError, match="unsupported claim level"):
        summarize_company_candidate(candidate)

    candidate = _candidate()
    candidate["business_observations"][0]["value"] = 0.9
    with pytest.raises(ValueError, match="value does not match numerator"):
        summarize_company_candidate(candidate)

    candidate = _candidate()
    candidate["business_observations"][0]["claim_level"] = "causal"
    with pytest.raises(ValueError, match="causal claims require experiment"):
        summarize_company_candidate(candidate)


def test_business_registry_is_unique_and_exposes_decision_questions():
    catalog = metric_catalog()
    ids = [metric["id"] for metric in BUSINESS_METRICS]

    assert len(ids) == len(set(ids))
    assert len(ids) >= 15
    assert all(metric["question"].endswith("?") for metric in BUSINESS_METRICS)
    assert catalog["principle"].startswith("quality, operating economics")


def test_v2_receipt_adapter_keeps_measured_and_judged_fields_explicit():
    receipt = {
        "schema_version": "aivideo-bench-candidate-execution-v2",
        "task_id": standard()["tasks"][0]["id"],
        "status": "completed",
        "latency_seconds": 3.0,
        "first_model_response_seconds": 0.5,
        "inference_seconds": 2.0,
        "mcp_seconds": 0.75,
        "inference_calls": 2,
        "token_usage": {
            "input_tokens": 100,
            "cached_input_tokens": 25,
            "reasoning_tokens": 10,
            "output_tokens": 30,
        },
        "tool_metrics": [
            {
                "name": "place_clips",
                "attempts": 2,
                "dispatched": 1,
                "successes": 1,
                "errors": 0,
                "invalid_arguments": 1,
                "recovered_errors": 1,
                "latency_seconds": 0.75,
            }
        ],
    }

    row = run_row_from_receipt(
        receipt,
        trial=1,
        annotations={
            "false_completion": True,
            "redundant_calls_by_tool": {"place_clips": 1},
        },
    )

    assert row["status"] == "completed"
    assert row["false_completion"] is True
    assert row["tool_metrics"][0]["redundant_calls"] == 1


def test_company_report_cli_writes_a_company_readable_report(tmp_path):
    candidate_path = tmp_path / "candidate.json"
    report_path = tmp_path / "company.md"
    candidate_path.write_text(json.dumps(_candidate()))

    exit_code = main(
        [
            "company-report",
            "--candidate",
            str(candidate_path),
            "--output",
            str(report_path),
        ]
    )

    assert exit_code == 0
    assert report_path.read_text().startswith("# AIVideo Agent Company Scorecard")
