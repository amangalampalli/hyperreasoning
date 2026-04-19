"""Build finalized search graph event payloads from task progress and results."""

from __future__ import annotations

from typing import Any

from backend.models import TaskRunResponse


class TaskSearchGraphCollector:
    def __init__(self, *, policy: str) -> None:
        self.policy = policy
        self._graph_events: list[dict[str, Any]] = []
        self._graph_entries: dict[str, Any] = {}
        self._graph_root_ids: list[str] = []
        self._graph_created_nodes: set[str] = set()
        self._graph_policy_run_id: str | None = None
        self._graph_sibling_info: dict[str, tuple[int, int]] = {}

    def on_progress(self, event: dict[str, Any]) -> None:
        event_name = str(event.get("event") or "")
        policy = event.get("policy") if isinstance(event.get("policy"), str) else self.policy
        if event_name == "search_started":
            self._emit_graph_started(policy, "Search graph derived from the completed run.")
            return
        if event_name == "plan_bank_built":
            plan_bank = event.get("plan_bank")
            if isinstance(plan_bank, dict):
                self._emit_plan_bank_graph(plan_bank, policy)
            return
        if event_name == "state_initialized":
            self._emit_initial_visible_state(event)
            return
        if event_name == "step_completed":
            self._emit_graph_step(event)

    def finalize(self, result: TaskRunResponse, *, policy: str | None = None) -> list[dict[str, Any]]:
        self._emit_graph_result(result, policy=policy or self.policy)
        return list(self._graph_events)

    def _append_graph_event(self, event: dict[str, Any]) -> None:
        self._graph_events.append(event)

    def _status_for_bank_result(
        self,
        *,
        compile_success: bool | None,
        visible_test_passed: bool | None,
        hidden_test_passed: bool | None,
    ) -> str:
        if visible_test_passed is True or hidden_test_passed is True:
            return "SUCCESS"
        if visible_test_passed is False:
            return "FAILED_TEST"
        if compile_success is False:
            return "FAILED_COMPILE"
        if compile_success is True:
            return "ACTIVE"
        return "ACTIVE"

    def _graph_run_id(self, policy: str | None) -> str:
        return policy or self.policy or "search"

    def _graph_title(self, policy: str | None) -> str:
        label = (policy or self.policy or "search").replace("_", " ").title()
        return f"{label} Search"

    def _plan_summary(self, plan: dict[str, Any] | None) -> str:
        if not isinstance(plan, dict):
            return ""
        strategy = str(plan.get("strategy") or "candidate").replace("_", " ")
        bugs = [str(item) for item in plan.get("suspected_bug_types") or []][:2]
        if bugs:
            return f"{strategy} | {', '.join(bugs)}"
        return strategy

    def _patch_summary(self, plan: dict[str, Any] | None) -> str:
        if not isinstance(plan, dict):
            return ""
        targets = [str(item) for item in plan.get("target_files") or []][:3]
        return f"Targets {', '.join(targets)}" if targets else "Patch candidate"

    def _dsl_summary(self, plan: dict[str, Any] | None) -> str:
        if not isinstance(plan, dict):
            return ""
        lines: list[str] = []
        strategy = plan.get("strategy")
        if strategy:
            lines.append(f"strategy={strategy}")
        target_files = [str(item) for item in plan.get("target_files") or []]
        if target_files:
            lines.append(f"files={', '.join(target_files)}")
        bug_types = [str(item) for item in plan.get("suspected_bug_types") or []]
        if bug_types:
            lines.append(f"bugs={', '.join(bug_types)}")
        validation_checks = [str(item) for item in plan.get("validation_checks") or []]
        if validation_checks:
            lines.append(f"checks={', '.join(validation_checks)}")
        notes = str(plan.get("notes") or "").strip()
        if notes:
            lines.append(f"notes={notes}")
        return "\n".join(lines)

    def _node_title(self, bank_id: str, plan: dict[str, Any] | None) -> str:
        if not isinstance(plan, dict):
            return bank_id
        return str(plan.get("strategy") or bank_id).replace("_", " ").title()

    def _ancestry_to_bank(self, bank_id: str) -> list[str]:
        result = ["root"]
        current: str | None = bank_id
        while current:
            result.append(current)
            entry = self._graph_entries.get(current)
            if not isinstance(entry, dict):
                break
            parent_bank_id = entry.get("parent_bank_id")
            current = str(parent_bank_id) if isinstance(parent_bank_id, str) and parent_bank_id else None
        result.reverse()
        if result and result[0] != "root":
            result.insert(0, "root")
        deduped: list[str] = []
        for node_id in result:
            if not deduped or deduped[-1] != node_id:
                deduped.append(node_id)
        return deduped

    def _emit_graph_started(self, policy: str | None, subtitle: str) -> None:
        run_id = self._graph_run_id(policy)
        self._graph_policy_run_id = run_id
        self._append_graph_event(
            {
                "type": "search_started",
                "run_id": run_id,
                "title": self._graph_title(policy),
                "subtitle": subtitle,
            }
        )

    def _emit_plan_bank_graph(self, plan_bank: dict[str, Any], policy: str | None) -> None:
        run_id = self._graph_run_id(policy)
        self._graph_entries = plan_bank.get("entries") if isinstance(plan_bank.get("entries"), dict) else {}
        self._graph_root_ids = [str(item) for item in plan_bank.get("root_bank_ids") or []]
        self._graph_created_nodes = {"root"}
        self._graph_sibling_info = {}
        self._append_graph_event(
            {
                "type": "node_created",
                "run_id": run_id,
                "node": {
                    "id": "root",
                    "title": "ROOT",
                    "status": "ROOT",
                    "createdOrder": 0,
                    "shortSummary": "Initial search state",
                    "dslSummary": "root_state",
                    "patchSummary": "Task root",
                },
            }
        )
        for index, bank_id in enumerate(self._graph_root_ids, start=1):
            self._graph_sibling_info[bank_id] = (index, len(self._graph_root_ids))
        for raw_entry in self._graph_entries.values():
            if not isinstance(raw_entry, dict):
                continue
            children = [str(item) for item in raw_entry.get("child_bank_ids") or []]
            for index, child_id in enumerate(children, start=1):
                self._graph_sibling_info[child_id] = (index, len(children))

    def _emit_graph_bank_node(self, bank_id: str) -> None:
        if bank_id in self._graph_created_nodes:
            return
        entry = self._graph_entries.get(bank_id)
        if not isinstance(entry, dict):
            return
        run_id = self._graph_policy_run_id or self._graph_run_id(self.policy)
        plan = entry.get("plan") if isinstance(entry.get("plan"), dict) else None
        child_index, child_count = self._graph_sibling_info.get(bank_id, (None, None))
        self._graph_created_nodes.add(bank_id)
        self._append_graph_event(
            {
                "type": "node_created",
                "run_id": run_id,
                "node": {
                    "id": bank_id,
                    "parentId": str(entry.get("parent_bank_id") or "root"),
                    "depth": int(entry.get("depth") or 0) + 1,
                    "title": self._node_title(bank_id, plan),
                    "shortSummary": self._plan_summary(plan),
                    "dslSummary": self._dsl_summary(plan),
                    "patchSummary": self._patch_summary(plan),
                    "rationaleSummary": str(plan.get("notes") or "") if isinstance(plan, dict) else "",
                    "childIndex": child_index,
                    "childCount": child_count,
                    "status": "IDLE",
                    "createdOrder": len(self._graph_created_nodes),
                    "rawMetadataJson": str(entry),
                },
            }
        )
        self._append_graph_event(
            {
                "type": "edge_created",
                "run_id": run_id,
                "parent_id": str(entry.get("parent_bank_id") or "root"),
                "child_id": bank_id,
            }
        )
        heuristic_score = entry.get("heuristic_score")
        if heuristic_score is not None:
            self._append_graph_event(
                {
                    "type": "node_scored",
                    "run_id": run_id,
                    "score": {
                        "id": bank_id,
                        "score": float(heuristic_score),
                        "heuristicScore": float(heuristic_score),
                    },
                }
            )

    def _emit_revealed_children(self, child_bank_ids: list[str]) -> None:
        for bank_id in child_bank_ids:
            self._emit_graph_bank_node(bank_id)

    def _emit_initial_visible_state(self, event: dict[str, Any]) -> None:
        visible_child_bank_ids = [
            str(item) for item in event.get("visible_child_bank_ids") or [] if isinstance(item, str)
        ]
        self._emit_revealed_children(visible_child_bank_ids)

    def _emit_graph_step(self, event: dict[str, Any]) -> None:
        run_id = self._graph_policy_run_id or self._graph_run_id(self.policy)
        action = event.get("action")
        visible_before = [
            str(item) for item in event.get("visible_child_bank_ids_before") or [] if isinstance(item, str)
        ]
        visible_after = [
            str(item) for item in event.get("visible_child_bank_ids_after") or [] if isinstance(item, str)
        ]
        self._emit_revealed_children(visible_before)
        self._emit_revealed_children(visible_after)

        current_bank_id = event.get("current_bank_id")
        is_structural_move = isinstance(action, str) and (
            action.startswith("SELECT_CHILD_")
            or action in {"REQUEST_MORE_CANDIDATES", "REFINE_CURRENT_PLAN", "BACKTRACK"}
        )
        if is_structural_move and isinstance(current_bank_id, str) and current_bank_id in self._graph_created_nodes:
            self._append_graph_event(
                {
                    "type": "node_status_changed",
                    "run_id": run_id,
                    "node_id": current_bank_id,
                    "status": "EXPANDING" if event.get("done") is not True else "ACTIVE",
                    "terminal_summary": self._describe_action(action),
                }
            )

        best_bank_id = event.get("best_bank_id")
        if isinstance(best_bank_id, str):
            self._append_graph_event(
                {
                    "type": "best_path_updated",
                    "run_id": run_id,
                    "node_ids": self._ancestry_to_bank(best_bank_id),
                }
            )
        elif isinstance(event.get("path_bank_ids"), list):
            path_ids = ["root", *[str(item) for item in event.get("path_bank_ids") or []]]
            self._append_graph_event(
                {
                    "type": "best_path_updated",
                    "run_id": run_id,
                    "node_ids": path_ids,
                }
            )

        if isinstance(current_bank_id, str) and any(
            event.get(key) is not None for key in ("compile_success", "visible_test_passed", "hidden_test_passed")
        ):
            self._append_graph_event(
                {
                    "type": "node_status_changed",
                    "run_id": run_id,
                    "node_id": current_bank_id,
                    "status": self._status_for_bank_result(
                        compile_success=event.get("compile_success"),
                        visible_test_passed=event.get("visible_test_passed"),
                        hidden_test_passed=event.get("hidden_test_passed"),
                    ),
                    "terminal_summary": self._describe_result(event),
                    "compile_status": self._compile_status(event),
                    "test_status": self._test_status(event),
                }
            )

    def _emit_graph_result(self, result: TaskRunResponse, *, policy: str | None) -> None:
        run_id = self._graph_run_id(policy)
        visited_bank_ids = {
            str(raw.get("bank_id") or (raw.get("state") or {}).get("current_bank_id"))
            for raw in result.nodes
            if isinstance(raw, dict)
            and isinstance(raw.get("bank_id") or (raw.get("state") or {}).get("current_bank_id"), str)
        }
        for bank_id in self._graph_entries.keys():
            if bank_id not in visited_bank_ids and bank_id in self._graph_created_nodes:
                self._append_graph_event(
                    {
                        "type": "node_pruned",
                        "run_id": run_id,
                        "node_id": bank_id,
                        "reason": "Never promoted into the executed branch",
                    }
                )

        if isinstance(result.strategy.best_bank_id, str):
            self._append_graph_event(
                {
                    "type": "best_path_updated",
                    "run_id": run_id,
                    "node_ids": self._ancestry_to_bank(result.strategy.best_bank_id),
                }
            )
            terminal_status = "SUCCESS" if result.strategy.visible_test_passed is True else (
                "FAILED_TEST" if result.strategy.visible_test_passed is False else (
                    "ACTIVE" if result.strategy.compile_success else "FAILED_COMPILE"
                )
            )
            self._append_graph_event(
                {
                    "type": "node_status_changed",
                    "run_id": run_id,
                    "node_id": result.strategy.best_bank_id,
                    "status": terminal_status,
                    "terminal_summary": self._result_summary(result),
                    "compile_status": (
                        "compiled"
                        if result.strategy.compile_success is True
                        else "compile failed"
                        if result.strategy.compile_success is False
                        else None
                    ),
                    "test_status": (
                        "visible tests passed"
                        if result.strategy.visible_test_passed is True
                        else "visible tests failed"
                        if result.strategy.visible_test_passed is False
                        else None
                    ),
                }
            )
        self._append_graph_event(
            {
                "type": "search_finished",
                "run_id": run_id,
                "terminal_node_id": result.strategy.best_bank_id,
                "success": bool(result.strategy.visible_test_passed is True or result.strategy.compile_success),
                "summary": self._result_summary(result),
            }
        )

    def _describe_action(self, action: Any) -> str | None:
        if not isinstance(action, str):
            return None
        if action.startswith("SELECT_CHILD_"):
            slot = action.rsplit("_", 1)[-1]
            try:
                return f"selected child {int(slot) + 1}"
            except ValueError:
                return action
        return action.replace("_", " ").lower()

    def _compile_status(self, event: dict[str, Any]) -> str | None:
        compile_success = event.get("compile_success")
        if compile_success is True:
            return "compiled"
        if compile_success is False:
            return "compile failed"
        return None

    def _test_status(self, event: dict[str, Any]) -> str | None:
        if event.get("hidden_test_passed") is True:
            return "hidden tests passed"
        if event.get("visible_test_passed") is True:
            return "visible tests passed"
        if event.get("visible_test_passed") is False:
            return "visible tests failed"
        return None

    def _describe_result(self, event: dict[str, Any]) -> str | None:
        if event.get("hidden_test_passed") is True:
            return "Hidden tests passed"
        if event.get("visible_test_passed") is True:
            return "Visible tests passed"
        if event.get("visible_test_passed") is False:
            return "Visible tests failed"
        if event.get("compile_success") is False:
            return "Compile failed"
        if event.get("compile_success") is True:
            return "Compiled"
        return self._describe_action(event.get("action"))

    def _result_summary(self, result: TaskRunResponse) -> str:
        if result.strategy.visible_test_passed is True:
            return "Visible tests passed for the best node."
        if result.strategy.visible_test_passed is False:
            return "Best node still fails visible tests."
        if result.strategy.compile_success:
            return "Best node compiled successfully."
        return "Search finished without a successful compile."
