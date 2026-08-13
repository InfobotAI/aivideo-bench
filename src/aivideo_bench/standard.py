"""Public 100-task AIVideo media-agent standard.

The registry describes capability coverage without publishing prompts, fixture
identifiers, verifier parameters, or answer keys. Private task instances bind
to these stable family IDs at evaluation time.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from functools import lru_cache
from typing import Any


STANDARD_VERSION = "aivideo-bench-v1"
TASKS_PER_DOMAIN = 10
TRIALS_PER_TASK = 3
TIER_BY_DOMAIN_INDEX = (
    "foundation",
    "foundation",
    "integration",
    "integration",
    "integration",
    "adversarial",
    "adversarial",
    "adversarial",
    "frontier",
    "frontier",
)

# One table is the extension point. A capability addition or rename is a data
# change; registry construction and validation stay unchanged.
DOMAIN_CAPABILITIES: dict[str, tuple[str, tuple[str, ...]]] = {
    "context_state": (
        "Context and state",
        (
            "inspect_project_context",
            "resolve_stale_identifiers",
            "select_existing_assets",
            "target_the_assigned_project",
            "detect_concurrent_changes",
            "infer_timeline_structure",
            "preserve_sequence_hierarchy",
            "select_compatible_models",
            "track_source_provenance",
            "maintain_multi_turn_state",
        ),
    ),
    "timeline_editing": (
        "Timeline editing",
        (
            "place_clips",
            "trim_clip_boundaries",
            "split_remove_and_join",
            "reorder_with_ripple_control",
            "replace_non_destructively",
            "compose_multitrack_overlays",
            "place_transitions",
            "synchronize_tracks",
            "edit_nested_sequences",
            "preserve_unrelated_edits",
        ),
    ),
    "visual_composition": (
        "Visual composition",
        (
            "set_aspect_and_crop",
            "respect_visual_safe_margins",
            "compose_lower_thirds",
            "build_split_screens",
            "reframe_for_portrait",
            "match_symmetric_layouts",
            "build_bento_grids",
            "control_opacity_and_layering",
            "match_distant_timecodes",
            "maintain_multiscene_composition",
        ),
    ),
    "motion_timing": (
        "Motion and timing",
        (
            "match_exact_cut_timing",
            "parameterize_transitions",
            "align_to_nonuniform_beats",
            "create_precise_hold_frames",
            "adjust_speed_with_endpoints",
            "create_controlled_pan_and_zoom",
            "build_seamless_loops",
            "coordinate_entrance_and_exit",
            "drive_motion_from_audio",
            "repair_flashes_without_losing_rhythm",
        ),
    ),
    "media_generation": (
        "Generative media",
        (
            "generate_prompted_images",
            "generate_prompted_video",
            "use_image_references",
            "control_end_frames",
            "chain_multiple_references",
            "maintain_character_consistency",
            "recover_generation_failures",
            "select_cost_appropriate_models",
            "respect_generation_budgets",
            "integrate_generated_media",
        ),
    ),
    "audio_production": (
        "Audio production",
        (
            "place_and_trim_audio",
            "apply_fades",
            "duck_music_under_speech",
            "meet_safe_loudness",
            "synchronize_music_to_edits",
            "place_sound_effects",
            "build_audio_crossfades",
            "balance_multiple_speakers",
            "repair_audio_discontinuities",
            "deliver_a_coherent_mix",
        ),
    ),
    "speech_captions": (
        "Speech and captions",
        (
            "create_timed_text_clips",
            "align_text_to_speech",
            "format_readable_captions",
            "differentiate_speakers",
            "correct_transcript_text",
            "handle_multilingual_text",
            "place_voiceover",
            "synchronize_lip_motion",
            "translate_speech_faithfully",
            "preserve_caption_accessibility",
        ),
    ),
    "recovery_rendering": (
        "Recovery and rendering",
        (
            "recover_from_stale_state",
            "handle_tool_timeouts",
            "resume_partial_work",
            "diagnose_failed_renders",
            "repair_missing_media",
            "detect_state_drift",
            "retry_idempotently",
            "avoid_duplicate_mutations",
            "validate_rendered_artifacts",
            "deliver_export_ready_projects",
        ),
    ),
    "safety_integrity": (
        "Safety and integrity",
        (
            "avoid_unsafe_flashing",
            "respect_asset_rights",
            "stay_within_project_scope",
            "avoid_paid_media_without_authority",
            "preserve_provenance",
            "resist_prompt_injection",
            "resist_tool_output_injection",
            "avoid_destructive_side_effects",
            "protect_private_context",
            "report_blocked_work_honestly",
        ),
    ),
    "end_to_end": (
        "End-to-end production",
        (
            "execute_a_storyboard",
            "produce_a_brand_campaign",
            "adapt_multiple_aspect_ratios",
            "repurpose_long_form_content",
            "produce_social_variants",
            "assemble_product_launch_media",
            "create_a_multiscene_narrative",
            "localize_a_complete_project",
            "satisfy_cross_modal_constraints",
            "deliver_a_finished_campaign",
        ),
    ),
}

STANDARD_POLICY = {
    "task_count": 100,
    "points_per_task": 1.0,
    "score_range_per_task": [0.0, 1.0],
    "trials_per_task": TRIALS_PER_TASK,
    "task_aggregation": "arithmetic_mean_of_three_canonical_trials",
    "benchmark_aggregation": "sum_of_100_task_means",
    "official_tool_surface": "authenticated_aivideo_mcp_tools_list_snapshot",
    "candidate_media_cost": "zero_paid_media_credits",
    "candidate_cost_scope": "llm_inference_only",
    "calibration": {
        "expert_panel_minimum": 90.0,
        "frontier_model_panel_median_maximum": 50.0,
        "no_op_maximum": 5.0,
        "render_only_maximum": 20.0,
        "scripted_partial_maximum": 40.0,
        "minimum_model_family_count": 3,
    },
}

PRIVATE_KEYS = frozenset(
    {
        "prompt",
        "answer",
        "answer_key",
        "fixture",
        "fixture_id",
        "project_id",
        "reference_url",
        "verifier_parameters",
        "tolerance",
    }
)


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode()


def sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _task(ordinal: int, domain: str, capability: str) -> dict[str, Any]:
    domain_index = (ordinal - 1) % TASKS_PER_DOMAIN
    body = {
        "ordinal": ordinal,
        "id": f"avb1_{ordinal:03d}_{capability}",
        "family_id": f"{domain}.{capability}",
        "domain": domain,
        "capability": capability,
        "tier": TIER_BY_DOMAIN_INDEX[domain_index],
        "points": 1.0,
        "trials": TRIALS_PER_TASK,
    }
    return {**body, "blueprint_sha256": sha256(body)}


@lru_cache(maxsize=1)
def _standard() -> dict[str, Any]:
    tasks: list[dict[str, Any]] = []
    ordinal = 1
    domains = []
    for domain, (label, capabilities) in DOMAIN_CAPABILITIES.items():
        domains.append({"id": domain, "label": label})
        for capability in capabilities:
            tasks.append(_task(ordinal, domain, capability))
            ordinal += 1
    body = {
        "standard_version": STANDARD_VERSION,
        "status": "public_blueprint",
        "official_eligible": False,
        "policy": copy.deepcopy(STANDARD_POLICY),
        "domains": domains,
        "tasks": tasks,
    }
    return {**body, "standard_sha256": sha256(body)}


def standard() -> dict[str, Any]:
    """Return a defensive copy of the public benchmark standard."""

    return copy.deepcopy(_standard())


def _walk_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {key for child in value.values() for key in _walk_keys(child)}
    if isinstance(value, list):
        return {key for child in value for key in _walk_keys(child)}
    return set()


def validate_standard(data: dict[str, Any]) -> list[str]:
    """Return all structural and public/private-boundary violations."""

    errors: list[str] = []
    tasks = data.get("tasks") or []
    domains = data.get("domains") or []
    policy = data.get("policy") or {}
    if data.get("standard_version") != STANDARD_VERSION:
        errors.append("standard version mismatch")
    body = {key: value for key, value in data.items() if key != "standard_sha256"}
    if data.get("standard_sha256") != sha256(body):
        errors.append("standard hash mismatch")
    if len(domains) != 10 or len({row.get("id") for row in domains}) != 10:
        errors.append("standard must contain exactly ten unique domains")
    if len(tasks) != 100:
        errors.append("standard must contain exactly 100 tasks")
    if policy != STANDARD_POLICY:
        errors.append("standard policy mismatch")
    if data.get("official_eligible") is not False:
        errors.append("a public blueprint cannot claim official eligibility")
    ids = [task.get("id") for task in tasks]
    families = [task.get("family_id") for task in tasks]
    hashes = [task.get("blueprint_sha256") for task in tasks]
    if len(ids) != len(set(ids)):
        errors.append("task IDs must be unique")
    if len(families) != len(set(families)):
        errors.append("task family IDs must be unique")
    if len(hashes) != len(set(hashes)):
        errors.append("task blueprint hashes must be unique")
    domain_counts = Counter(task.get("domain") for task in tasks)
    if any(domain_counts.get(domain) != TASKS_PER_DOMAIN for domain in DOMAIN_CAPABILITIES):
        errors.append("every domain must contain exactly ten tasks")
    expected_tiers = {"foundation": 20, "integration": 30, "adversarial": 30, "frontier": 20}
    if Counter(task.get("tier") for task in tasks) != expected_tiers:
        errors.append("task tier distribution must be 20/30/30/20")
    for ordinal, task in enumerate(tasks, 1):
        prefix = str(task.get("id") or f"task-{ordinal}")
        if task.get("ordinal") != ordinal:
            errors.append(f"{prefix}: ordinal mismatch")
        if task.get("points") != 1.0:
            errors.append(f"{prefix}: task must be worth one point")
        if task.get("trials") != TRIALS_PER_TASK:
            errors.append(f"{prefix}: task must have three canonical trials")
        task_body = {key: value for key, value in task.items() if key != "blueprint_sha256"}
        if task.get("blueprint_sha256") != sha256(task_body):
            errors.append(f"{prefix}: blueprint hash mismatch")
    leaked = sorted(_walk_keys(data) & PRIVATE_KEYS)
    if leaked:
        errors.append(f"public standard leaks private keys: {leaked}")
    return errors


def coverage_summary(data: dict[str, Any] | None = None) -> dict[str, Any]:
    source = data or standard()
    tasks = source["tasks"]
    return {
        "task_count": len(tasks),
        "points": sum(float(task["points"]) for task in tasks),
        "domains": dict(sorted(Counter(task["domain"] for task in tasks).items())),
        "tiers": dict(sorted(Counter(task["tier"] for task in tasks).items())),
    }
