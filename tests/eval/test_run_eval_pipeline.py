from __future__ import annotations

import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval.run_eval_pipeline import (
    build_pairwise_summary,
    canonical_method_name,
    choose_raw_output_file_mode,
    build_search_config,
    compute_paired_deltas,
    exact_mcnemar_p_value,
    exact_sign_flip_p_value,
    load_completed_run_keys,
    merge_preserved_fields,
    normalize_raw_object,
    remove_method_records,
)


def test_canonical_method_name_handles_aliases() -> None:
    assert canonical_method_name("rainbow") == "rainbow"
    assert canonical_method_name("oneshot") == "one_shot"
    assert canonical_method_name("one-shot") == "one_shot"
    assert canonical_method_name("one_shot") == "one_shot"


def test_normalize_task_eval_record_fills_total_tokens_and_fraction() -> None:
    normalized = normalize_raw_object(
        {
            "kind": "task_eval_record_v1",
            "task_id": "task_a",
            "method": "one_shot",
            "solved": 1,
            "tests_passed": 3,
            "tests_total": 4,
            "elapsed_time_ms": 12.5,
            "llm_input_tokens": 10,
            "llm_output_tokens": 7,
            "branches_explored": 2,
            "steps_to_success": 2,
            "seed": 123,
        }
    )

    assert normalized is not None
    assert normalized["method"] == "one_shot"
    assert normalized["llm_total_tokens"] == 17
    assert math.isclose(normalized["fraction_tests_passed"], 0.75)


def test_exact_mcnemar_p_value_for_two_discordant_pairs() -> None:
    value = exact_mcnemar_p_value(
        rainbow=[1, 1, 0, 0],  # type: ignore[arg-type]
        baseline=[0, 0, 0, 0],  # type: ignore[arg-type]
    )
    assert math.isclose(value, 0.5)


def test_exact_sign_flip_p_value_small_exact_case() -> None:
    value = exact_sign_flip_p_value(deltas=[1.0, 1.0, 1.0])  # type: ignore[arg-type]
    assert math.isclose(value, 0.25)


def test_compute_paired_deltas_pairs_tasks_and_emits_summary() -> None:
    rows = [
        {
            "task_id": "task_a",
            "method": "rainbow",
            "solved": 1,
            "tests_passed": 4,
            "tests_total": 4,
            "fraction_tests_passed": 1.0,
            "elapsed_time_ms": 100.0,
            "llm_input_tokens": 10,
            "llm_output_tokens": 5,
            "llm_total_tokens": 15,
            "branches_explored": 2,
            "steps_to_success": 2,
            "seed": 123,
        },
        {
            "task_id": "task_a",
            "method": "heuristic",
            "solved": 0,
            "tests_passed": 1,
            "tests_total": 4,
            "fraction_tests_passed": 0.25,
            "elapsed_time_ms": 140.0,
            "llm_input_tokens": 8,
            "llm_output_tokens": 4,
            "llm_total_tokens": 12,
            "branches_explored": 3,
            "steps_to_success": None,
            "seed": 123,
        },
        {
            "task_id": "task_b",
            "method": "rainbow",
            "solved": 1,
            "tests_passed": 4,
            "tests_total": 4,
            "fraction_tests_passed": 1.0,
            "elapsed_time_ms": 90.0,
            "llm_input_tokens": 9,
            "llm_output_tokens": 5,
            "llm_total_tokens": 14,
            "branches_explored": 2,
            "steps_to_success": 2,
            "seed": 123,
        },
        {
            "task_id": "task_b",
            "method": "heuristic",
            "solved": 1,
            "tests_passed": 3,
            "tests_total": 4,
            "fraction_tests_passed": 0.75,
            "elapsed_time_ms": 130.0,
            "llm_input_tokens": 8,
            "llm_output_tokens": 4,
            "llm_total_tokens": 12,
            "branches_explored": 3,
            "steps_to_success": 3,
            "seed": 123,
        },
    ]

    paired = compute_paired_deltas(rows, resamples=200, seed=123)
    solve_row = next(row for row in paired if row["baseline"] == "heuristic" and row["metric"] == "solved")

    assert solve_row["n_tasks"] == 2
    assert math.isclose(solve_row["mean_delta"], 0.5)
    assert "solve rate" in solve_row["summary"].lower()
    assert build_pairwise_summary(solve_row).startswith("Rainbow")


def test_load_completed_run_keys_reads_existing_records(tmp_path: Path) -> None:
    raw = tmp_path / "raw_eval_results.jsonl"
    raw.write_text(
        "\n".join(
            [
                '{"kind":"task_eval_record_v1","task_id":"task_a","method":"rainbow","seed":123}',
                '{"kind":"task_eval_record_v1","task_id":"task_b","method":"oneshot","seed":123}',
                '{"kind":"task_eval_error_v1","task_id":"task_c","method":"heuristic","seed":123,"error":"boom"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    completed = load_completed_run_keys(raw)

    assert ("task_a", "rainbow", 123) in completed
    assert ("task_b", "one_shot", 123) in completed
    assert ("task_c", "heuristic", 123) not in completed


def test_remove_method_records_prunes_only_requested_methods(tmp_path: Path) -> None:
    raw = tmp_path / "raw_eval_results.jsonl"
    raw.write_text(
        "\n".join(
            [
                '{"kind":"task_eval_record_v1","task_id":"task_a","method":"rainbow","seed":123}',
                '{"kind":"task_eval_record_v1","task_id":"task_a","method":"heuristic","seed":123}',
                '{"kind":"task_eval_record_v1","task_id":"task_b","method":"one_shot","seed":123}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    remove_method_records(raw, {"rainbow"})
    contents = raw.read_text(encoding="utf-8")

    assert '"method":"rainbow"' not in contents
    assert '"method":"heuristic"' in contents
    assert '"method":"one_shot"' in contents


def test_merge_preserved_fields_overrides_only_requested_fields() -> None:
    new_record = {
        "task_id": "task_a",
        "method": "rainbow",
        "solved": 1,
        "tests_passed": 4,
        "elapsed_time_ms": 200.0,
        "llm_total_tokens": 300,
    }
    existing_record = {
        "task_id": "task_a",
        "method": "rainbow",
        "solved": 0,
        "tests_passed": 1,
        "elapsed_time_ms": 999.0,
        "llm_total_tokens": 777,
    }

    merged = merge_preserved_fields(
        new_record,
        existing_record,
        {"elapsed_time_ms", "llm_total_tokens"},
    )

    assert merged["solved"] == 1
    assert merged["tests_passed"] == 4
    assert merged["elapsed_time_ms"] == 999.0
    assert merged["llm_total_tokens"] == 777


def test_choose_raw_output_file_mode_appends_for_targeted_reruns() -> None:
    assert choose_raw_output_file_mode(
        raw_output_exists=True,
        resume=False,
        replace_methods={"rainbow"},
        preserve_fields=set(),
    ) == "a"
    assert choose_raw_output_file_mode(
        raw_output_exists=True,
        resume=False,
        replace_methods=set(),
        preserve_fields={"elapsed_time_ms"},
    ) == "a"
    assert choose_raw_output_file_mode(
        raw_output_exists=True,
        resume=False,
        replace_methods=set(),
        preserve_fields=set(),
    ) == "w"


def test_one_shot_eval_config_uses_deterministic_temperatures() -> None:
    class Args:
        max_steps = 8
        proposal_source = "heuristic"
        python_bin = "python"
        timeout_s = 12.0
        compiler_temp = 0.2
        allow_full_file_fallback = False
        max_verified_plans_per_task = 1
        seed = 123

    one_shot_config = build_search_config("one_shot", Args())
    rainbow_config = build_search_config("rainbow", Args())

    assert one_shot_config.proposal_source == "llm"
    assert one_shot_config.llm_proposal_temperature == 0.0
    assert one_shot_config.compiler_temp == 0.0
    assert rainbow_config.llm_proposal_temperature is None
    assert rainbow_config.compiler_temp == 0.2

