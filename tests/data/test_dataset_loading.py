from __future__ import annotations

from pathlib import Path
import json

from data.datasets import load_transition_file


def test_load_transition_file_from_run_dir(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    sample = {
        "state": {"task_id": "task-1", "family": "family", "current_strategy": "ROOT", "valid_actions": ["TERMINATE"], "child_slots": []},
        "action": "TERMINATE",
        "reward": 1.0,
        "next_state": None,
        "done": True,
        "valid_actions": ["TERMINATE"],
        "info": {},
    }
    (run_dir / "dataset.jsonl").write_text(json.dumps(sample) + "\n", encoding="utf-8")
    transitions = load_transition_file(run_dir)
    assert len(transitions) == 1
    assert transitions[0].action == "TERMINATE"
