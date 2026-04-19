from __future__ import annotations

from pathlib import Path

import orjson

from env.dsl_schema import PlanDSL
from llm.compiler import _apply_edit_operations, _repair_compiler_output, compile_plan_to_code
from llm.prompt_utils import load_task_context


def _write_task(task_dir: Path) -> None:
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "build_graph.py").write_text(
        "\n".join(
            [
                "def build_graph(edges):",
                "    graph = {}",
                "    for left, right in edges:",
                "        graph.setdefault(left, []).append(right)",
                "    return graph",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (task_dir / "test_visible.py").write_text(
        "\n".join(
            [
                "import unittest",
                "from build_graph import build_graph",
                "",
                "class BuildGraphTests(unittest.TestCase):",
                "    def test_basic(self):",
                "        self.assertEqual(build_graph([(1, 2)]), {1: [2]})",
                "",
                "if __name__ == '__main__':",
                "    unittest.main()",
                "",
            ]
        ),
        encoding="utf-8",
    )
    payload = {
        "task_id": "incremental_build_graph_bug_test",
        "family": "incremental_build_graph_bug",
        "difficulty": "easy",
        "language": "python",
        "prompt": "Fix graph construction behavior.",
        "target_files": ["build_graph.py"],
        "visible_test_file": "test_visible.py",
        "metadata": {"target_files": ["build_graph.py"]},
    }
    (task_dir / "task.json").write_bytes(orjson.dumps(payload))


class _FakeLLM:
    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []

    def complete(self, prompt: str, **_: object) -> str:
        self.prompts.append(prompt)
        return self.response


def test_apply_edit_operations_accepts_dedent_and_whitespace_tolerant_match(tmp_path: Path) -> None:
    _write_task(tmp_path)
    task = load_task_context(tmp_path)

    compiled = _apply_edit_operations(
        task,
        [
            {
                "file": "build_graph.py",
                "old_snippet": "\n".join(
                    [
                        "graph = {}",
                        "for left, right in edges:",
                        "    graph.setdefault(left, []).append(right)",
                        "return graph",
                    ]
                ),
                "new_snippet": "\n".join(
                    [
                        "graph = {}",
                        "for left, right in edges:",
                        "    graph.setdefault(left, []).append(right)",
                        "    graph.setdefault(right, [])",
                        "return graph",
                    ]
                ),
            }
        ],
    )

    assert "build_graph.py" in compiled
    assert "graph.setdefault(right, [])" in compiled["build_graph.py"]


def test_apply_edit_operations_accepts_whitespace_insensitive_signature_match(tmp_path: Path) -> None:
    task_dir = tmp_path
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "client.py").write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "",
                "def list_ids(",
                "    records: list[dict[str, object]],",
                "    *,",
                "    include_inactive: bool = False,",
                "    min_score: int = 0,",
                ") -> list[str]:",
                "    return []",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (task_dir / "task.json").write_bytes(
        orjson.dumps(
            {
                "task_id": "signature_match_test",
                "family": "multi_file_interface_drift",
                "difficulty": "easy",
                "language": "python",
                "prompt": "Fix signature formatting drift.",
                "target_files": ["client.py"],
                "metadata": {"target_files": ["client.py"]},
            }
        )
    )
    task = load_task_context(task_dir)

    compiled = _apply_edit_operations(
        task,
        [
            {
                "file": "client.py",
                "old_snippet": "\n".join(
                    [
                        "def list_ids(",
                        "    records: list[dict[str, object]],",
                        "    *, ",
                        "    include_inactive: bool = False,",
                        "    min_score: int = 0,",
                        ")",
                        " -> list[str]:",
                    ]
                ),
                "new_snippet": "\n".join(
                    [
                        "def list_ids(",
                        "    records: list[dict[str, object]],",
                        "    *,",
                        "    include_inactive: bool = False,",
                        "    min_score: int = 0,",
                        ") -> list[str]:",
                    ]
                ),
            }
        ],
    )

    assert ") -> list[str]:" in compiled["client.py"]


def test_apply_edit_operations_ignores_duplicate_noop_edit(tmp_path: Path) -> None:
    task_dir = tmp_path
    task_dir.mkdir(parents=True, exist_ok=True)
    client_source = "\n".join(
        [
            "from __future__ import annotations",
            "",
            "def list_ids(",
            "    records: list[dict[str, object]],",
            "    *,",
            "    include_inactive: bool = False,",
            "    min_score: int = 0,",
            ") -> list[str]:",
            "    summary = summarize_records(",
            "        records,",
            "        include_inactive=include_inactive,",
            "        min_score=min_score,",
            "    )",
            "    return list(summary[\"ids\"])",
            "",
            "def average_score(",
            "    records: list[dict[str, object]],",
            "    *,",
            "    include_inactive: bool = False,",
            "    min_score: int = 0,",
            ") -> float:",
            "    summary = summarize_records(",
            "        records,",
            "        include_inactive=include_inactive,",
            "        min_score=min_score,",
            "    )",
            "    return 0.0",
            "",
        ]
    )
    (task_dir / "client.py").write_text(client_source, encoding="utf-8")
    (task_dir / "task.json").write_bytes(
        orjson.dumps(
            {
                "task_id": "noop_duplicate_test",
                "family": "multi_file_interface_drift",
                "difficulty": "easy",
                "language": "python",
                "prompt": "Ignore no-op duplicates.",
                "target_files": ["client.py"],
                "metadata": {"target_files": ["client.py"]},
            }
        )
    )
    task = load_task_context(task_dir)

    compiled = _apply_edit_operations(
        task,
        [
            {
                "file": "client.py",
                "old_snippet": "\n".join(
                    [
                        "    summary = summarize_records(",
                        "        records,",
                        "        include_inactive=include_inactive,",
                        "        min_score=min_score,",
                        "    )",
                    ]
                ),
                "new_snippet": "\n".join(
                    [
                        "    summary = summarize_records(",
                        "        records,",
                        "        include_inactive=include_inactive,",
                        "        min_score=min_score,",
                        "    )",
                    ]
                ),
            }
        ],
    )

    assert compiled["client.py"] == client_source


def test_repair_compiler_output_uses_real_target_filename() -> None:
    client = _FakeLLM('{"build_graph.py": "print(1)"}')
    repaired = _repair_compiler_output(
        client,
        raw_output="bad output",
        target_files=["build_graph.py"],
        want_edits=False,
    )

    assert repaired == '{"build_graph.py": "print(1)"}'
    assert '"build_graph.py"' in client.prompts[0]
    assert '"target.py"' not in client.prompts[0]


def _single_target_plan(task) -> PlanDSL:
    return PlanDSL.from_dict(
        {
            "strategy": "rebuild_propagation_fix",
            "target_files": ["build_graph.py"],
            "suspected_bug_types": ["missing_transitive_rebuild"],
            "invariants": ["all downstream nodes rebuild"],
            "subgoals": ["repair affected traversal"],
            "validation_checks": ["visible_tests"],
            "risks": ["generic"],
            "touched_symbols": ["build_graph"],
            "edit_style": "surgical_patch",
            "confidence": 0.6,
            "notes": "repair build graph",
        },
        task_id=task.task_id,
        family=task.family,
        language=task.language,
        task_target_files=task.target_files,
    )


def test_compile_plan_recovers_single_target_placeholder_filename(tmp_path: Path) -> None:
    _write_task(tmp_path)
    task = load_task_context(tmp_path)
    plan = _single_target_plan(task)
    client = _FakeLLM('{"target.py": "def build_graph(edges):\\n    return {}\\n"}')

    compiled = compile_plan_to_code(task, plan, client=client, allow_full_file_fallback=True)

    assert compiled == {"build_graph.py": "def build_graph(edges):\n    return {}\n"}


def test_compile_plan_recovers_single_target_content_key(tmp_path: Path) -> None:
    _write_task(tmp_path)
    task = load_task_context(tmp_path)
    plan = _single_target_plan(task)
    client = _FakeLLM('{"content": "def build_graph(edges):\\n    return {}\\n"}')

    compiled = compile_plan_to_code(task, plan, client=client, allow_full_file_fallback=True)

    assert compiled == {"build_graph.py": "def build_graph(edges):\n    return {}\n"}


def test_compile_plan_recovers_single_target_code_fence(tmp_path: Path) -> None:
    _write_task(tmp_path)
    task = load_task_context(tmp_path)
    plan = _single_target_plan(task)
    client = _FakeLLM("```python\ndef build_graph(edges):\n    return {}\n```")

    compiled = compile_plan_to_code(task, plan, client=client, allow_full_file_fallback=True)

    assert compiled == {"build_graph.py": "def build_graph(edges):\n    return {}\n"}
