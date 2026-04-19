"""Multi-file interface drift task family."""

from __future__ import annotations

from data.task_templates.base import TaskSpec, TaskTemplate
from data.task_templates.utils import build_rng, choose_variant, dedent, make_task_id, render_test_module


class MultiFileInterfaceDriftTemplate(TaskTemplate):
    """Generate coordinated multi-file contract repair tasks."""

    family = "multi_file_interface_drift"

    def generate_instance(self, seed: int, difficulty: str) -> TaskSpec:
        self._validate_difficulty(difficulty)
        rng = build_rng(seed, self.family, difficulty)
        api_name = choose_variant(rng, ["summarize_records", "build_summary"])
        prompt = choose_variant(
            rng,
            [
                f"Repair the public summary contract implemented across `api.py` and `client.py`. The "
                f"`{api_name}` API should accept `include_inactive` and `min_score`, and it must return a "
                "dictionary with `ids`, `count`, and `total_score`. The helper functions in `client.py` must "
                "stay consistent with that contract.",
                f"`api.py` and `client.py` drifted during a refactor. Restore one stable interface for "
                f"`{api_name}` and the client helpers without changing the tests.",
            ],
        )
        reference_api = dedent(
            f"""
            from __future__ import annotations


            def {api_name}(
                records: list[dict[str, object]],
                *,
                include_inactive: bool = False,
                min_score: int = 0,
            ) -> dict[str, object]:
                ids: list[str] = []
                total_score = 0
                for record in records:
                    is_active = bool(record.get("active", True))
                    if not include_inactive and not is_active:
                        continue
                    score = int(record["score"])
                    if score < min_score:
                        continue
                    ids.append(str(record["id"]))
                    total_score += score
                return {{
                    "ids": ids,
                    "count": len(ids),
                    "total_score": total_score,
                }}
            """
        )
        reference_client = dedent(
            f"""
            from __future__ import annotations

            from api import {api_name}


            def list_ids(
                records: list[dict[str, object]],
                *,
                include_inactive: bool = False,
                min_score: int = 0,
            ) -> list[str]:
                summary = {api_name}(
                    records,
                    include_inactive=include_inactive,
                    min_score=min_score,
                )
                return list(summary["ids"])


            def average_score(
                records: list[dict[str, object]],
                *,
                include_inactive: bool = False,
                min_score: int = 0,
            ) -> float:
                summary = {api_name}(
                    records,
                    include_inactive=include_inactive,
                    min_score=min_score,
                )
                if summary["count"] == 0:
                    return 0.0
                return summary["total_score"] / summary["count"]
            """
        )

        if difficulty == "medium":
            buggy_api = reference_api.replace('"ids": ids,\n                    "count": len(ids),\n                    "total_score": total_score,\n', '"items": ids,\n                    "count": len(ids),\n                    "total": total_score,\n')
            buggy_client = reference_client
            bug_types = ["api return keys drifted away from the documented client contract"]
            strategy_traps = [
                "Fixing only one call site is insufficient because hidden tests hit the API directly",
                "A compatibility shim must preserve the documented keys and defaults consistently",
            ]
        else:
            buggy_api = reference_api.replace(
                "                include_inactive: bool = False,\n                min_score: int = 0,\n",
                "                include_disabled: bool = False,\n                min_score: int = 0,\n",
            ).replace(
                "                    if not include_inactive and not is_active:\n",
                "                    if not include_disabled and not is_active:\n",
            ).replace(
                '"ids": ids,\n                    "count": len(ids),\n                    "total_score": total_score,\n',
                '"items": ids,\n                    "count": len(ids),\n                    "total": total_score,\n',
            )
            buggy_client = reference_client.replace(
                "                    include_inactive=include_inactive,\n",
                "                    include_disabled=include_inactive,\n",
            ).replace(
                '                return list(summary["ids"])\n',
                '                return list(summary["items"])\n',
            ).replace(
                '                if summary["count"] == 0:\n                    return 0.0\n                return summary["total_score"] / summary["count"]\n',
                '                if summary["count"] == 0:\n                    return 0.0\n                return summary["total"] / summary["count"]\n',
            )
            bug_types = ["api and client both use a stale pre-refactor interface shape"]
            strategy_traps = [
                "Visible client behavior can look fine while the public API still violates the documented contract",
                "Restoring only api.py or only client.py leaves direct callers and helpers out of sync",
            ]

        visible_tests = render_test_module(
            dedent(
                """
                from client import average_score, list_ids


                RECORDS = [
                    {"id": "a", "score": 5, "active": True},
                    {"id": "b", "score": 2, "active": False},
                    {"id": "c", "score": 8, "active": True},
                ]


                class InterfaceVisibleTests(unittest.TestCase):
                    def test_list_ids_filters_inactive_by_default(self) -> None:
                        self.assertEqual(list_ids(RECORDS), ["a", "c"])

                    def test_average_score_respects_threshold(self) -> None:
                        self.assertEqual(average_score(RECORDS, min_score=6), 8.0)
                """
            )
        )

        hidden_tests = render_test_module(
            dedent(
                f"""
                from api import {api_name}
                from client import average_score, list_ids


                RECORDS = [
                    {{"id": "a", "score": 5, "active": True}},
                    {{"id": "b", "score": 2, "active": False}},
                    {{"id": "c", "score": 8, "active": True}},
                ]


                class InterfaceHiddenTests(unittest.TestCase):
                    def test_api_uses_documented_keyword_and_keys(self) -> None:
                        summary = {api_name}(RECORDS, include_inactive=True)
                        self.assertEqual(summary, {{"ids": ["a", "b", "c"], "count": 3, "total_score": 15}})

                    def test_empty_result_uses_zero_average(self) -> None:
                        self.assertEqual(average_score(RECORDS, min_score=99), 0.0)

                    def test_client_and_api_stay_consistent(self) -> None:
                        summary = {api_name}(RECORDS, include_inactive=False, min_score=5)
                        self.assertEqual(list_ids(RECORDS, min_score=5), summary["ids"])
                """
            )
        )

        return self.build_spec(
            seed=seed,
            difficulty=difficulty,
            prompt=prompt,
            files={
                "api.py": buggy_api,
                "client.py": buggy_client,
                "test_visible.py": visible_tests,
                "test_hidden.py": hidden_tests,
            },
            reference_files={"api.py": reference_api, "client.py": reference_client},
            entrypoint="api.py",
            visible_test_file="test_visible.py",
            hidden_test_file="test_hidden.py",
            task_id=make_task_id(self.family, seed),
            metadata={
                "bug_type": bug_types,
                "strategy_traps": strategy_traps,
                "target_files": ["api.py", "client.py"],
                "expected_skill_tags": ["interfaces", "multi-file", "contracts", "refactors"],
                "niche_topic": "post-refactor interface drift",
                "repairable": True,
            },
        )
