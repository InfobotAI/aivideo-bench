import copy
import json

import pytest

from aivideo_bench.cli import main
from aivideo_bench.company import (
    BUSINESS_METRICS,
    COMPANY_INPUT_SCHEMA_VERSION,
    build_company_report,
    build_run_manifest,
    metric_catalog,
    render_company_markdown,
    run_row_from_receipt,
    summarize_company_candidate,
)
from aivideo_bench.standard import sha256, standard


PACK_SHA = "a" * 64
TOOL_SURFACE_SHA = "b" * 64
TASK_PACK_ID_SHA = "1" * 64
PROMPT_POLICY_SHA = "c" * 64
VERIFIER_POLICY_SHA = "d" * 64
VERIFIER_POLICY_VERSION = "company-verifier-v1"
MANIFEST_KEY = b"k" * 32


def _config(model):
    return {
        "model_id": model,
        "exact_model_id": model,
        "provider_slug": "provider-a",
        "reasoning_effort": "medium",
    }


def _identity(model):
    return sha256(_config(model))


def _result_rows(*, quality=0.8, operational=True, cost=0.01):
    return [
        {
            "task_id": task["id"],
            "trial": trial,
            "quality_score": quality,
            "operational_success": operational
            and not (task_index == 0 and trial == 1),
            "exact_success": quality == 1.0,
            "execution_valid": True,
            "cost_cap_respected": True,
            "llm_inference_cost_usd": cost,
            "paid_media_credits": 0.0,
        }
        for task_index, task in enumerate(standard()["tasks"])
        for trial in range(1, 4)
    ]


def _run_rows(model, *, pack_sha=PACK_SHA, tool_surface_sha=TOOL_SURFACE_SHA):
    config_sha = _identity(model)
    rows = []
    for task_index, task in enumerate(standard()["tasks"]):
        for trial in range(1, 4):
            first = task_index == 0 and trial == 1
            rows.append(
                {
                    "task_id": task["id"],
                    "trial": trial,
                    "receipt_sha256": sha256(
                        {"task_id": task["id"], "trial": trial, "model": model}
                    ),
                    "pack_sha256": pack_sha,
                    "tool_surface_sha256": tool_surface_sha,
                    "candidate_config_sha256": config_sha,
                    "prompt_policy_sha256": PROMPT_POLICY_SHA,
                    "verifier_policy_version": VERIFIER_POLICY_VERSION,
                    "verifier_policy_sha256": VERIFIER_POLICY_SHA,
                    "verifier_evidence_sha256": sha256(
                        {"task_id": task["id"], "trial": trial, "reviewed": True}
                    ),
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
                    "llm_inference_cost_usd": 0.01,
                    "paid_media_credits": 0.0,
                    "cost_cap_respected": True,
                    "tool_metrics": [
                        {
                            "name": "place_clips",
                            "attempts": 3 if first else 2,
                            "dispatched": 2,
                            "successes": 1,
                            "errors": 1,
                            "invalid_arguments": 1 if first else 0,
                            "redundant_calls": 0,
                            "same_tool_successes_after_error": 2 if first else 1,
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


def _business_observation(config_sha, *, claim="observed"):
    evidence = {
        "cohort_id": "paid-agent-users-july",
        "query_sha256": "e" * 64,
        "exposure_identity_sha256": config_sha,
    }
    source = "production_telemetry"
    if claim == "directional":
        evidence.update(
            {
                "comparator_numerator": 60,
                "comparator_denominator": 100,
                "comparator_window": "2026-06-01/2026-06-30",
                "comparator_query_sha256": "f" * 64,
                "comparator_exposure_identity_sha256": "6" * 64,
            }
        )
    elif claim == "causal":
        source = "experiment"
        evidence = {
            "experiment_id": "agent-model-rct-1",
            "assignment_unit": "user",
            "control_numerator": 60,
            "control_denominator": 100,
            "analysis_policy_version": "intent-to-treat-v1",
            "analysis_artifact_sha256": "5" * 64,
            "exposure_identity_sha256": config_sha,
            "control_exposure_identity_sha256": "4" * 64,
            "effect_ci_lower": 0.02,
            "effect_ci_upper": 0.22,
        }
    return {
        "metric_id": "task_to_usable_project_rate",
        "value": 0.72,
        "numerator": 72,
        "denominator": 100,
        "window": "2026-07-01/2026-07-31",
        "claim_level": claim,
        "source": source,
        "evidence": evidence,
    }


def _candidate(
    model="Candidate A",
    *,
    business=True,
    pack_sha=PACK_SHA,
    tool_surface_sha=TOOL_SURFACE_SHA,
    task_pack_identity_sha=TASK_PACK_ID_SHA,
    comparison_kind="standard",
    variant_id=None,
    ablation_group_id=None,
    calibration_passed=True,
):
    results = _result_rows()
    runs = _run_rows(model, pack_sha=pack_sha, tool_surface_sha=tool_surface_sha)
    bindings = [
        {
            "task_id": row["task_id"],
            "trial": row["trial"],
            "receipt_sha256": row["receipt_sha256"],
        }
        for row in runs
    ]
    config_sha = _identity(model)
    business_rows = [_business_observation(config_sha)] if business else []
    manifest = build_run_manifest(
        variant_id=variant_id or model.lower().replace(" ", "-"),
        comparison_kind=comparison_kind,
        ablation_group_id=ablation_group_id,
        task_pack_identity_sha256=task_pack_identity_sha,
        pack_sha256=pack_sha,
        tool_surface_sha256=tool_surface_sha,
        candidate_config=_config(model),
        prompt_policy_sha256=PROMPT_POLICY_SHA,
        verifier_policy_version=VERIFIER_POLICY_VERSION,
        verifier_policy_sha256=VERIFIER_POLICY_SHA,
        calibration_gate={
            "policy_version": "frontier-panel-v1",
            "evidence_sha256": "2" * 64,
            "passed": calibration_passed,
        },
        result_rows=results,
        run_rows=runs,
        result_receipt_bindings=bindings,
        business_observations=business_rows,
        key_id="company-manifest-test-key",
        signing_key=MANIFEST_KEY,
    )
    return {
        "schema_version": COMPANY_INPUT_SCHEMA_VERSION,
        "manifest": manifest,
        "result_rows": results,
        "run_rows": runs,
        "result_receipt_bindings": bindings,
        "business_observations": business_rows,
    }


def _refresh_manifest(candidate):
    old = candidate["manifest"]
    candidate["manifest"] = build_run_manifest(
        variant_id=old["variant_id"],
        comparison_kind=old["comparison_kind"],
        ablation_group_id=old["ablation_group_id"],
        task_pack_identity_sha256=old["task_pack_identity_sha256"],
        pack_sha256=old["pack_sha256"],
        tool_surface_sha256=old["tool_surface_sha256"],
        candidate_config=old["candidate_config"],
        prompt_policy_sha256=old["prompt_policy_sha256"],
        verifier_policy_version=old["verifier_policy_version"],
        verifier_policy_sha256=old["verifier_policy_sha256"],
        calibration_gate=old["calibration_gate"],
        result_rows=candidate["result_rows"],
        run_rows=candidate["run_rows"],
        result_receipt_bindings=candidate["result_receipt_bindings"],
        business_observations=candidate["business_observations"],
        key_id=old["attestation"]["key_id"],
        signing_key=MANIFEST_KEY,
    )


def test_company_summary_keeps_evidence_planes_separate_and_bound():
    summary = summarize_company_candidate(
        _candidate(), manifest_signing_key=MANIFEST_KEY
    )

    assert summary["quality"]["official_score"] == 80.0
    assert summary["speed"]["end_to_end_seconds"] == {
        "mean": 2.0,
        "p50": 2.0,
        "p95": 2.0,
    }
    assert summary["economics"]["llm_inference_usd"] == 3.0
    assert summary["economics"]["cost_per_operational_success_usd"] == 0.010033
    assert summary["economics"]["tokens"]["reasoning_tokens"] == 3000
    assert summary["tooling"]["attempts"] == 601
    assert summary["tooling"]["dispatch_success_rate"] == 0.5
    assert summary["tooling"]["same_tool_success_after_error_rate"] == 1.0
    assert summary["reliability"]["false_completion_rate"] == 0.003333
    assert summary["business"]["measured_metric_count"] == 1
    assert summary["manifest"]["candidate_config_sha256"] == _identity("Candidate A")


def test_report_refuses_composite_and_marks_missing_business_data():
    report = build_company_report(
        [_candidate("Candidate A"), _candidate("Candidate B", business=False)],
        manifest_signing_key=MANIFEST_KEY,
    )
    rendered = render_company_markdown(report)

    assert report["comparison_policy"]["overall_composite"] is None
    assert "There is intentionally no opaque overall winner score" in rendered
    assert "Business impact is not included in the 100 points" in rendered
    assert "72.0% (n=100, observed)" in rendered
    assert "not measured" in rendered
    assert "Same-tool success after error" in rendered


def test_runtime_matrix_must_be_complete_unique_and_manifest_bound():
    candidate = _candidate()
    candidate["run_rows"] = candidate["run_rows"][:-1]
    _refresh_manifest(candidate)
    with pytest.raises(ValueError, match="exactly 300 canonical cells"):
        summarize_company_candidate(candidate, manifest_signing_key=MANIFEST_KEY)

    candidate = _candidate()
    candidate["run_rows"][-1] = copy.deepcopy(candidate["run_rows"][0])
    _refresh_manifest(candidate)
    with pytest.raises(ValueError, match="duplicate result cell"):
        summarize_company_candidate(candidate, manifest_signing_key=MANIFEST_KEY)

    candidate = _candidate()
    candidate["run_rows"][0]["output_tokens"] += 1
    with pytest.raises(ValueError, match="runtime matrix hash"):
        summarize_company_candidate(candidate, manifest_signing_key=MANIFEST_KEY)


def test_execution_identity_cannot_be_mixed_across_matrices():
    candidate = _candidate()
    candidate["run_rows"][0]["candidate_config_sha256"] = _identity("Other Model")
    _refresh_manifest(candidate)

    with pytest.raises(ValueError, match="candidate_config_sha256 differs"):
        summarize_company_candidate(candidate, manifest_signing_key=MANIFEST_KEY)

    candidate = _candidate()
    candidate["result_receipt_bindings"][0]["receipt_sha256"] = "7" * 64
    _refresh_manifest(candidate)
    with pytest.raises(ValueError, match="result and runtime receipt identities differ"):
        summarize_company_candidate(candidate, manifest_signing_key=MANIFEST_KEY)

    other_pack = "9" * 64
    with pytest.raises(ValueError, match="do not share pack and tool surface"):
        build_company_report(
            [_candidate("Candidate A"), _candidate("Candidate B", pack_sha=other_pack)],
            manifest_signing_key=MANIFEST_KEY,
        )


def test_manifest_is_authenticated_and_model_identity_is_derived():
    candidate = _candidate()
    candidate["manifest"]["variant_id"] = "tampered"
    with pytest.raises(ValueError, match="manifest HMAC mismatch"):
        summarize_company_candidate(candidate, manifest_signing_key=MANIFEST_KEY)

    candidate = _candidate()
    candidate["manifest"]["candidate_config"]["exact_model_id"] = "other-model"
    with pytest.raises(ValueError, match="manifest HMAC mismatch"):
        summarize_company_candidate(candidate, manifest_signing_key=MANIFEST_KEY)


def test_official_score_requires_cost_and_calibration_gates():
    summary = summarize_company_candidate(
        _candidate(calibration_passed=False), manifest_signing_key=MANIFEST_KEY
    )
    assert summary["quality"]["official_score"] is None
    assert summary["quality"]["score_status"] == "calibration_gate_failed"
    assert summary["quality"]["diagnostic_quality_score"] == 80.0

    candidate = _candidate()
    candidate["result_rows"][0]["cost_cap_respected"] = False
    candidate["run_rows"][0]["cost_cap_respected"] = False
    _refresh_manifest(candidate)
    summary = summarize_company_candidate(
        candidate, manifest_signing_key=MANIFEST_KEY
    )
    assert summary["quality"]["official_score"] is None
    assert summary["quality"]["score_status"] == "cost_gate_failed"

    candidate = _candidate()
    candidate["run_rows"][0]["llm_inference_cost_usd"] = 0.02
    _refresh_manifest(candidate)
    with pytest.raises(ValueError, match="runtime and result inference cost differ"):
        summarize_company_candidate(candidate, manifest_signing_key=MANIFEST_KEY)


def test_runtime_status_and_annotations_must_agree_with_scored_result():
    candidate = _candidate()
    candidate["run_rows"][1]["status"] = "tool_policy_violation"
    _refresh_manifest(candidate)
    with pytest.raises(ValueError, match="non-completed runtime"):
        summarize_company_candidate(candidate, manifest_signing_key=MANIFEST_KEY)

    candidate = _candidate()
    candidate["run_rows"][1]["destructive_side_effect"] = True
    _refresh_manifest(candidate)
    with pytest.raises(ValueError, match="catastrophic runtime annotation"):
        summarize_company_candidate(candidate, manifest_signing_key=MANIFEST_KEY)

    candidate = _candidate()
    candidate["run_rows"][0]["status"] = "tool_policy_violation"
    candidate["result_rows"][0]["operational_success"] = False
    candidate["result_rows"][0]["exact_success"] = False
    _refresh_manifest(candidate)
    with pytest.raises(ValueError, match="contradicts execution validity"):
        summarize_company_candidate(candidate, manifest_signing_key=MANIFEST_KEY)

    candidate = _candidate()
    candidate["run_rows"][0]["status"] = "unknown_status"
    _refresh_manifest(candidate)
    with pytest.raises(ValueError, match="status is not canonical"):
        summarize_company_candidate(candidate, manifest_signing_key=MANIFEST_KEY)


def test_consistent_policy_violation_preserves_diagnostics_but_blocks_score():
    candidate = _candidate()
    candidate["run_rows"][0]["status"] = "tool_policy_violation"
    candidate["result_rows"][0]["operational_success"] = False
    candidate["result_rows"][0]["exact_success"] = False
    candidate["result_rows"][0]["execution_valid"] = False
    _refresh_manifest(candidate)

    summary = summarize_company_candidate(
        candidate, manifest_signing_key=MANIFEST_KEY
    )

    assert summary["quality"]["official_score"] is None
    assert summary["quality"]["diagnostic_quality_score"] == 80.0


def test_cost_preflight_block_preserves_zero_spend_but_blocks_official_score():
    candidate = _candidate()
    candidate["run_rows"][0]["status"] = "cost_preflight_blocked"
    candidate["run_rows"][0]["llm_inference_cost_usd"] = 0.0
    candidate["run_rows"][0]["cost_cap_respected"] = False
    candidate["result_rows"][0]["llm_inference_cost_usd"] = 0.0
    candidate["result_rows"][0]["cost_cap_respected"] = False
    _refresh_manifest(candidate)

    summary = summarize_company_candidate(
        candidate, manifest_signing_key=MANIFEST_KEY
    )

    assert summary["economics"]["llm_inference_usd"] == 2.99
    assert summary["quality"]["official_score"] is None
    assert summary["quality"]["score_status"] == "cost_gate_failed"
    assert summary["quality"]["diagnostic_quality_score"] == 80.0

    candidate = _candidate()
    candidate["run_rows"][0]["status"] = "cost_preflight_blocked"
    _refresh_manifest(candidate)
    with pytest.raises(ValueError, match="cost-gate-failed runtime"):
        summarize_company_candidate(candidate, manifest_signing_key=MANIFEST_KEY)


def test_same_model_tool_ablation_requires_explicit_matched_group():
    base = _candidate(
        "Candidate A", variant_id="with-tools", ablation_group_id="mcp-value-1"
    )
    ablation = _candidate(
        "Candidate A",
        variant_id="without-tools",
        comparison_kind="tool_ablation",
        ablation_group_id="mcp-value-1",
        pack_sha="7" * 64,
        tool_surface_sha="8" * 64,
    )

    report = build_company_report(
        [base, ablation], manifest_signing_key=MANIFEST_KEY
    )
    assert [row["model"] for row in report["candidates"]] == [
        "Candidate A",
        "Candidate A",
    ]
    assert len({row["variant_id"] for row in report["candidates"]}) == 2

    unrelated = _candidate(
        "Candidate A",
        variant_id="unrelated-pack",
        comparison_kind="tool_ablation",
        ablation_group_id="mcp-value-1",
        pack_sha="7" * 64,
        tool_surface_sha="8" * 64,
        task_pack_identity_sha="9" * 64,
    )
    with pytest.raises(ValueError, match="do not share evaluation policy"):
        build_company_report([base, unrelated], manifest_signing_key=MANIFEST_KEY)


def test_standard_comparability_is_independent_of_candidate_order():
    base = _candidate(
        "Candidate A", variant_id="with-tools", ablation_group_id="mcp-value-1"
    )
    ablation = _candidate(
        "Candidate A",
        variant_id="without-tools",
        comparison_kind="tool_ablation",
        ablation_group_id="mcp-value-1",
        pack_sha="7" * 64,
        tool_surface_sha="8" * 64,
    )
    incompatible_standard = _candidate(
        "Candidate B",
        variant_id="different-standard-surface",
        pack_sha="9" * 64,
        tool_surface_sha="6" * 64,
    )

    with pytest.raises(ValueError, match="do not share pack and tool surface"):
        build_company_report(
            [ablation, base, incompatible_standard],
            manifest_signing_key=MANIFEST_KEY,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("inference_seconds", 3.0, "inference_seconds exceeds"),
        ("mcp_seconds", 3.0, "mcp_seconds exceeds"),
        ("inference_seconds", 1.75, "inference plus MCP"),
    ],
)
def test_runtime_rejects_impossible_component_timing(field, value, message):
    candidate = _candidate()
    candidate["run_rows"][0][field] = value
    if field == "mcp_seconds":
        candidate["run_rows"][0]["tool_metrics"][0]["latency_seconds"] = value
    _refresh_manifest(candidate)

    with pytest.raises(ValueError, match=message):
        summarize_company_candidate(candidate, manifest_signing_key=MANIFEST_KEY)


def test_runtime_reconciles_per_tool_and_total_mcp_timing():
    candidate = _candidate()
    candidate["run_rows"][0]["tool_metrics"][0]["latency_seconds"] = 0.25
    _refresh_manifest(candidate)

    with pytest.raises(ValueError, match="per-tool latency differs"):
        summarize_company_candidate(candidate, manifest_signing_key=MANIFEST_KEY)


def test_tool_accounting_rejects_inconsistent_data():
    candidate = _candidate()
    candidate["run_rows"][0]["tool_metrics"][0]["successes"] = 2
    _refresh_manifest(candidate)
    with pytest.raises(ValueError, match="successes plus errors must equal dispatched"):
        summarize_company_candidate(candidate, manifest_signing_key=MANIFEST_KEY)


def test_business_claims_require_claim_specific_evidence_and_exposure():
    candidate = _candidate()
    candidate["business_observations"][0]["value"] = 0.9
    _refresh_manifest(candidate)
    with pytest.raises(ValueError, match="value does not match numerator"):
        summarize_company_candidate(candidate, manifest_signing_key=MANIFEST_KEY)

    candidate = _candidate()
    candidate["business_observations"] = [
        _business_observation(_identity("Candidate A"), claim="directional")
    ]
    _refresh_manifest(candidate)
    summary = summarize_company_candidate(candidate, manifest_signing_key=MANIFEST_KEY)
    evidence = summary["business"]["observations"][
        "task_to_usable_project_rate"
    ]["evidence"]
    assert evidence["comparator_value"] == 0.6
    assert evidence["absolute_delta"] == 0.12

    candidate = _candidate()
    candidate["business_observations"] = [
        _business_observation(_identity("Candidate A"), claim="causal")
    ]
    _refresh_manifest(candidate)
    summary = summarize_company_candidate(candidate, manifest_signing_key=MANIFEST_KEY)
    evidence = summary["business"]["observations"][
        "task_to_usable_project_rate"
    ]["evidence"]
    assert evidence["control_value"] == 0.6
    assert evidence["absolute_effect"] == 0.12

    candidate["business_observations"][0]["source"] = "production_telemetry"
    _refresh_manifest(candidate)
    with pytest.raises(ValueError, match="causal claims require experiment"):
        summarize_company_candidate(candidate, manifest_signing_key=MANIFEST_KEY)

    candidate = _candidate()
    observation = _business_observation(
        _identity("Candidate A"), claim="directional"
    )
    observation["evidence"]["comparator_numerator"] = 150
    candidate["business_observations"] = [observation]
    _refresh_manifest(candidate)
    with pytest.raises(ValueError, match="comparator rate exceeds 1"):
        summarize_company_candidate(candidate, manifest_signing_key=MANIFEST_KEY)

    candidate = _candidate()
    observation = _business_observation(_identity("Candidate A"), claim="causal")
    observation["evidence"]["effect_ci_lower"] = 0.2
    candidate["business_observations"] = [observation]
    _refresh_manifest(candidate)
    with pytest.raises(ValueError, match="effect is outside confidence interval"):
        summarize_company_candidate(candidate, manifest_signing_key=MANIFEST_KEY)

    candidate = _candidate()
    candidate["business_observations"][0]["evidence"][
        "exposure_identity_sha256"
    ] = _identity("Other Model")
    _refresh_manifest(candidate)
    with pytest.raises(ValueError, match="business exposure differs"):
        summarize_company_candidate(candidate, manifest_signing_key=MANIFEST_KEY)


def test_business_registry_is_unique_and_data_driven():
    catalog = metric_catalog()
    ids = [metric["id"] for metric in BUSINESS_METRICS]

    assert len(ids) == len(set(ids))
    assert len(ids) >= 15
    assert all(metric["question"].endswith("?") for metric in BUSINESS_METRICS)
    assert catalog["principle"].startswith("quality, operating economics")


def _receipt():
    return {
        "schema_version": "aivideo-bench-candidate-execution-v2",
        "task_id": standard()["tasks"][0]["id"],
        "pack_sha256": PACK_SHA,
        "tool_surface_sha256": TOOL_SURFACE_SHA,
        "candidate": _config("Candidate A"),
        "prompt_policy_sha256": PROMPT_POLICY_SHA,
        "status": "completed",
        "latency_seconds": 3.0,
        "first_model_response_seconds": 0.5,
        "inference_seconds": 2.0,
        "mcp_seconds": 0.75,
        "inference_calls": 2,
        "llm_inference_cost_usd": 0.01,
        "paid_media_credits": 0.0,
        "cost_cap_respected": True,
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
                "same_tool_successes_after_error": 1,
                "latency_seconds": 0.75,
            }
        ],
    }


def _annotations():
    return {
        "verifier_policy_version": VERIFIER_POLICY_VERSION,
        "verifier_policy_sha256": VERIFIER_POLICY_SHA,
        "verifier_evidence_sha256": "8" * 64,
        "human_interventions": 0,
        "false_completion": True,
        "destructive_side_effect": False,
        "context_limit_hit": False,
        "stuck_loop": False,
        "external_dependency_failure": False,
        "redundant_calls_by_tool": {"place_clips": 1},
    }


def test_receipt_adapter_requires_complete_review_provenance():
    row = run_row_from_receipt(_receipt(), trial=1, annotations=_annotations())

    assert row["false_completion"] is True
    assert row["tool_metrics"][0]["redundant_calls"] == 1
    assert row["candidate_config_sha256"] == _identity("Candidate A")
    assert row["receipt_sha256"] == sha256(_receipt())

    notes = _annotations()
    del notes["false_completion"]
    with pytest.raises(ValueError, match="not complete and canonical"):
        run_row_from_receipt(_receipt(), trial=1, annotations=notes)

    notes = _annotations()
    notes["redundant_calls_by_tool"] = {"unknown_tool": 0}
    with pytest.raises(ValueError, match="exactly match receipt tools"):
        run_row_from_receipt(_receipt(), trial=1, annotations=notes)


def test_company_report_cli_writes_a_company_readable_report(tmp_path, monkeypatch):
    candidate_path = tmp_path / "candidate.json"
    report_path = tmp_path / "company.md"
    candidate_path.write_text(json.dumps(_candidate()))
    monkeypatch.setenv("AIVIDEO_BENCH_MANIFEST_SIGNING_KEY", MANIFEST_KEY.decode())

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
