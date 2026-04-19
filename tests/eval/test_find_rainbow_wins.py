from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval.find_rainbow_wins import (
    canonical_method_name,
    default_records_output_path,
    evaluate_records,
    find_rainbow_edges,
    render_markdown,
)


def _row(task_id: str, method: str, solved: int, tests: int, *, tokens: int = 100, time: float = 100.0) -> dict:
    return {
        "kind": "task_eval_record_v1",
        "task_id": task_id,
        "family": "family",
        "method": method,
        "solved": solved,
        "tests_passed": tests,
        "tests_total": 4,
        "fraction_tests_passed": tests / 4,
        "llm_total_tokens": tokens,
        "elapsed_time_ms": time,
        "branches_explored": 2,
    }


def test_find_rainbow_edges_identifies_strict_and_any_wins() -> None:
    rows = [
        _row("strict", "rainbow", 1, 4, tokens=10, time=10),
        _row("strict", "heuristic", 0, 0, tokens=20, time=20),
        _row("strict", "one_shot", 0, 0, tokens=30, time=30),
        _row("any", "rainbow", 1, 4, tokens=10, time=10),
        _row("any", "heuristic", 1, 4, tokens=20, time=20),
        _row("any", "one_shot", 0, 2, tokens=30, time=30),
        _row("tie_eff", "rainbow", 1, 4, tokens=10, time=10),
        _row("tie_eff", "heuristic", 1, 4, tokens=20, time=20),
        _row("tie_eff", "one_shot", 1, 4, tokens=30, time=30),
    ]

    report = find_rainbow_edges(rows, competitors=("heuristic", "one_shot"))

    assert [item["task_id"] for item in report["strict_solve_wins"]] == ["strict"]
    assert [item["task_id"] for item in report["any_solve_wins"]] == ["any", "strict"]
    assert [item["task_id"] for item in report["test_fraction_wins"]] == ["strict"]
    assert [item["task_id"] for item in report["quality_tie_efficiency_wins"]] == ["tie_eff"]


def test_canonical_method_name_normalizes_one_shot_alias() -> None:
    assert canonical_method_name("oneshot") == "one_shot"
    assert canonical_method_name("one-shot") == "one_shot"


def test_render_markdown_includes_task_and_count() -> None:
    report = find_rainbow_edges(
        [
            _row("strict", "rainbow", 1, 4),
            _row("strict", "heuristic", 0, 0),
        ],
        competitors=("heuristic",),
    )

    rendered = render_markdown(report, competitors=("heuristic",))

    assert "Count: 1" in rendered
    assert "strict" in rendered


def test_default_records_output_is_local_hyper_jsonl() -> None:
    path = default_records_output_path()

    assert ".hyper" in path.parts
    assert "rainbow_edge_scans" in path.parts
    assert path.name.startswith("fresh_eval_records_")
    assert path.suffix == ".jsonl"


def test_evaluate_records_parallel_preserves_run_order(monkeypatch) -> None:
    import argparse
    import scripts.eval.find_rainbow_wins as module

    class FakeTask:
        def __init__(self, task_id: str) -> None:
            self.task_id = task_id
            self.family = "family"

    class FakeStore:
        @classmethod
        def from_manifest(cls, manifest_path, limit=None):
            return cls()

        def iter_contexts(self):
            return [FakeTask("task_a"), FakeTask("task_b")]

    class InlineFuture:
        def __init__(self, value):
            self._value = value

        def result(self):
            return self._value

    class InlineExecutor:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def submit(self, fn, *args):
            return InlineFuture(fn(*args))

    def fake_worker(index, task, method, args):
        return index, {
            "kind": "task_eval_record_v1",
            "task_id": task.task_id,
            "family": task.family,
            "method": method,
            "solved": 1,
        }

    monkeypatch.setitem(__import__("sys").modules, "data.task_store", type("M", (), {"TaskStore": FakeStore}))
    monkeypatch.setattr(module, "ProcessPoolExecutor", InlineExecutor)
    monkeypatch.setattr(module, "as_completed", lambda futures: list(reversed(futures)))
    monkeypatch.setattr(module, "_evaluate_task_method_worker", fake_worker)

    args = argparse.Namespace(
        competitors=("heuristic", "one_shot"),
        task_dirs=None,
        task_manifest="manifest",
        num_tasks=None,
        no_progress=True,
        jobs=2,
        checkpoint="checkpoint",
    )

    records = evaluate_records(args)

    assert [(row["task_id"], row["method"]) for row in records] == [
        ("task_a", "rainbow"),
        ("task_a", "heuristic"),
        ("task_a", "one_shot"),
        ("task_b", "rainbow"),
        ("task_b", "heuristic"),
        ("task_b", "one_shot"),
    ]
