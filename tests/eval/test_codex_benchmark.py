from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval.codex_benchmark_lib import (
    build_codex_command,
    extract_usage_fields,
    load_rainbow_baseline_rows,
    parse_codex_jsonl_output,
    summarize_codex_events,
    CODEX_TIERS,
)


def test_parse_codex_jsonl_output_ignores_warning_lines() -> None:
    stdout = "\n".join(
        [
            "Reading additional input from stdin...",
            '{"type":"thread.started","thread_id":"abc"}',
            '{"type":"turn.completed","usage":{"input_tokens":10,"output_tokens":5}}',
        ]
    )

    events, warnings = parse_codex_jsonl_output(stdout)

    assert len(events) == 2
    assert warnings == ["Reading additional input from stdin..."]


def test_extract_usage_fields_total_only() -> None:
    usage = {"input_tokens": 100, "output_tokens": 25, "total_tokens": 125}

    extracted = extract_usage_fields(usage)

    assert extracted["llm_input_tokens"] == 100
    assert extracted["llm_output_tokens"] == 25
    assert extracted["llm_total_tokens"] == 125
    assert extracted["llm_reasoning_tokens"] is None
    assert extracted["llm_execution_tokens"] is None


def test_extract_usage_fields_reasoning_breakdown() -> None:
    usage = {
        "input_tokens": 100,
        "output_tokens": 40,
        "total_tokens": 140,
        "output_tokens_details": {"reasoning_tokens": 12},
    }

    extracted = extract_usage_fields(usage)

    assert extracted["llm_reasoning_tokens"] == 12
    assert extracted["llm_execution_tokens"] == 28


def test_summarize_codex_events_pulls_usage_and_last_message() -> None:
    events = [
        {"type": "thread.started", "thread_id": "abc"},
        {"type": "item.completed", "item": {"type": "agent_message", "text": "Done"}},
        {"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3}},
    ]

    summary, usage = summarize_codex_events(events, ["warn"])

    assert summary["event_counts"]["thread.started"] == 1
    assert summary["last_agent_message"] == "Done"
    assert summary["warning_count"] == 1
    assert usage == {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3}


def test_build_codex_command_includes_reasoning_tier() -> None:
    command = build_codex_command(
        workspace=Path("/tmp/workspace"),
        prompt="Fix it",
        tier=CODEX_TIERS[0],
        output_last_message=Path("/tmp/last.txt"),
        model="gpt-5.4",
    )

    assert "codex" == command[0]
    assert "--json" in command
    assert '--skip-git-repo-check' in command
    assert any('model_reasoning_effort="low"' == part for part in command)


def test_load_rainbow_baseline_rows_recomputes_solved_from_all_tests(tmp_path: Path) -> None:
    raw = tmp_path / "raw.jsonl"
    raw.write_text(
        '\n'.join(
            [
                '{"kind":"task_eval_record_v1","task_id":"task_a","method":"rainbow","tests_passed":3,"tests_total":3,"fraction_tests_passed":1.0,"elapsed_time_ms":10,"llm_total_tokens":20,"seed":123}',
                '{"kind":"task_eval_record_v1","task_id":"task_b","method":"rainbow","tests_passed":1,"tests_total":2,"fraction_tests_passed":0.5,"elapsed_time_ms":10,"llm_total_tokens":20,"seed":123}',
            ]
        )
        + '\n',
        encoding="utf-8",
    )

    rows = load_rainbow_baseline_rows(raw, task_ids={"task_a", "task_b"}, seed=123)

    by_task = {row["task_id"]: row for row in rows}
    assert by_task["task_a"]["solved"] == 1
    assert by_task["task_b"]["solved"] == 0
