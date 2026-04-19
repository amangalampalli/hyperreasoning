from __future__ import annotations

from pathlib import Path

import orjson

from env.dsl_schema import PlanDSL
from env.dsl_env import ACTION_SPACE, SearchControlConfig, collect_task_dataset
from env.verifier import CachedTaskVerifier
from llm.prompt_utils import load_task_context


def _write_task(task_dir: Path) -> None:
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "retry.py").write_text(
        "\n".join(
            [
                "def run_with_retry(fn, retries=1):",
                "    try:",
                "        return fn()",
                "    except Exception:",
                "        if retries <= 0:",
                "            raise",
                "        return fn()",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (task_dir / "test_visible.py").write_text(
        "\n".join(
            [
                "import unittest",
                "from retry import run_with_retry",
                "",
                "class RetryTests(unittest.TestCase):",
                "    def test_success(self):",
                "        self.assertEqual(run_with_retry(lambda: 7), 7)",
                "",
                "if __name__ == '__main__':",
                "    unittest.main()",
                "",
            ]
        ),
        encoding="utf-8",
    )
    payload = {
        "task_id": "async_retry_contract_test",
        "family": "async_retry_contract",
        "difficulty": "easy",
        "language": "python",
        "prompt": "Fix retry behavior without changing the public API.",
        "target_files": ["retry.py"],
        "visible_test_file": "test_visible.py",
        "metadata": {"target_files": ["retry.py"]},
    }
    (task_dir / "task.json").write_bytes(orjson.dumps(payload))


def test_cached_task_verifier_reuses_compile_result(tmp_path: Path) -> None:
    _write_task(tmp_path)
    task = load_task_context(tmp_path)
    plan = PlanDSL.from_dict(
        {
            "strategy": "minimal_patch",
            "target_files": ["retry.py"],
            "suspected_bug_types": ["cancellation_swallowing"],
            "invariants": ["cancelled calls are never retried"],
            "subgoals": ["split cancellation path"],
            "validation_checks": ["visible_tests"],
            "risks": ["generic"],
            "touched_symbols": ["run_with_retry"],
            "edit_style": "surgical_patch",
            "confidence": 0.7,
            "notes": "patch retry guard",
        },
        task_id=task.task_id,
        family=task.family,
        language=task.language,
        task_target_files=task.target_files,
    )

    compile_calls = {"count": 0}

    def fake_compile(task_context, plan, **kwargs):
        compile_calls["count"] += 1
        return {"retry.py": "def run_with_retry(fn, retries=1):\n    return fn()\n"}

    verifier = CachedTaskVerifier(
        run_tests=False,
        max_verified_plans=2,
        compile_fn=fake_compile,
    )

    first = verifier.verify(task, plan)
    second = verifier.verify(task, plan)

    assert compile_calls["count"] == 1
    assert first.compile_success is True
    assert second.cached is True
    assert second.plan_signature == first.plan_signature


def test_collect_task_dataset_generates_control_episodes(tmp_path: Path) -> None:
    _write_task(tmp_path)
    task = load_task_context(tmp_path)

    compile_calls = {"count": 0}

    def fake_compile(task_context, plan, **kwargs):
        compile_calls["count"] += 1
        return {"retry.py": "def run_with_retry(fn, retries=1):\n    return fn()\n"}

    verifier = CachedTaskVerifier(
        run_tests=False,
        max_verified_plans=3,
        compile_fn=fake_compile,
    )
    config = SearchControlConfig(
        episodes_per_task=4,
        max_steps_per_episode=5,
        max_bank_depth=2,
        max_root_plans=4,
        proposal_source="heuristic",
        run_tests=False,
        max_verified_plans_per_task=3,
        seed=7,
    )

    bundle = collect_task_dataset(task, config, verifier=verifier)

    assert bundle.summary["episodes"] == 4
    assert bundle.summary["transitions"] == len(bundle.transitions)
    assert bundle.summary["bank_root_plans"] > 0
    assert bundle.summary["bank_total_plans"] >= bundle.summary["bank_root_plans"]
    assert compile_calls["count"] <= 3
    assert any(transition.action in ACTION_SPACE for transition in bundle.transitions)
    assert all(transition.action in ACTION_SPACE for transition in bundle.transitions)
    assert any(
        transition.info.get("label_tier") in {"compile", "visible_test"} for transition in bundle.transitions
    )
