from __future__ import annotations

from pathlib import Path

import orjson

from env.dsl_env import DSLSearchEnv


def _write_task(task_dir: Path) -> None:
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "retry.py").write_text("def run_with_retry(fn, retries=1):\n    return fn()\n", encoding="utf-8")
    (task_dir / "test_visible.py").write_text(
        "import unittest\nfrom retry import run_with_retry\n\nclass T(unittest.TestCase):\n    def test_ok(self):\n        self.assertEqual(run_with_retry(lambda: 1), 1)\n\nif __name__ == '__main__':\n    unittest.main()\n",
        encoding="utf-8",
    )
    (task_dir / "task.json").write_bytes(
        orjson.dumps(
            {
                "task_id": "async_retry_contract_test",
                "family": "async_retry_contract",
                "difficulty": "easy",
                "language": "python",
                "prompt": "Fix retry behavior",
                "target_files": ["retry.py"],
                "visible_test_file": "test_visible.py",
                "metadata": {"target_files": ["retry.py"]},
            }
        )
    )


def test_env_reset_and_step(tmp_path: Path) -> None:
    _write_task(tmp_path / "tasks" / "async_retry_contract_test")
    env = DSLSearchEnv(tasks_root=tmp_path / "tasks")
    obs, info = env.reset()
    assert info["action_mask"].any()
    action = env.heuristic_action()
    next_obs, reward, terminated, truncated, step_info = env.step(action)
    assert isinstance(reward, float)
    assert truncated is False
    assert "action_mask" in step_info
    assert next_obs is None or isinstance(next_obs, dict)
    assert isinstance(terminated, bool)
