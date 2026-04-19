from __future__ import annotations

from fastapi.testclient import TestClient

from backend.api import create_app


def test_health_endpoint() -> None:
    app = create_app("http://127.0.0.1:8080")
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"


def test_task_run_request_accepts_oneshot_policy() -> None:
    from backend.models import TaskRunRequest

    request = TaskRunRequest(
        prompt="fix this",
        files={"foo.py": "print('x')\n"},
        target_files=["foo.py"],
        policy="oneshot",
    )

    assert request.policy == "oneshot"


def test_compare_request_defaults_include_oneshot() -> None:
    from backend.models import CompareStrategiesRequest

    request = CompareStrategiesRequest(
        prompt="fix this",
        files={"foo.py": "print('x')\n"},
        target_files=["foo.py"],
    )

    assert request.policies == ["heuristic", "rainbow", "oneshot"]


def test_oneshot_search_config_is_constrained() -> None:
    from backend.models import TaskRunRequest
    from backend.service import _build_search_config

    request = TaskRunRequest(
        prompt="fix this",
        files={"foo.py": "print('x')\n"},
        target_files=["foo.py"],
        policy="oneshot",
        proposal_source="hybrid",
        max_steps=8,
        max_verified_plans_per_task=6,
    )

    config = _build_search_config(request)

    assert config.max_steps_per_episode == 2
    assert config.max_bank_depth == 0
    assert config.root_candidate_batches == 1
    assert config.root_candidates_per_batch == 1
    assert config.max_root_plans == 1
    assert config.initial_root_reveal == 1
    assert config.proposal_source == "llm"
    assert config.max_verified_plans_per_task == 1


def test_compare_endpoint_schema(monkeypatch) -> None:
    from backend import service

    def fake_compare(request, *, llm_base_url: str):
        from backend.models import CompareStrategiesResponse, StrategySummary, TaskRunResponse

        strategy = StrategySummary(
            policy="heuristic",
            task_id="plugin_task",
            family="custom_single_file",
            total_reward=1.0,
            steps=3,
            compile_successes=1,
            visible_passes=1,
        )
        return CompareStrategiesResponse(
            strategies=[
                TaskRunResponse(
                    strategy=strategy,
                    root_candidates=[],
                    nodes=[],
                    edges=[],
                    transitions=[],
                )
            ]
        )

    monkeypatch.setattr(service, "compare_strategies", fake_compare)
    app = create_app("http://127.0.0.1:8080")
    client = TestClient(app)
    response = client.post(
        "/api/task/compare",
        json={
            "prompt": "fix this",
            "files": {"foo.py": "print('x')\n"},
            "target_files": ["foo.py"],
            "policies": ["heuristic"],
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["strategies"][0]["strategy"]["policy"] == "heuristic"


def test_run_endpoint_updates_progress_bar(monkeypatch) -> None:
    from backend import service

    progress_events: list[tuple[str, object]] = []

    class FakeTqdm:
        def __init__(
            self,
            total: int,
            desc: str,
            unit: str,
            leave: bool,
            position: int = 0,
            dynamic_ncols: bool = False,
        ) -> None:
            self.total = total
            self.desc = desc
            self.unit = unit
            self.leave = leave
            self.position = position
            self.dynamic_ncols = dynamic_ncols
            self.n = 0

        def update(self, amount: int) -> None:
            self.n += amount
            progress_events.append(("update", amount))

        def set_postfix(self, **kwargs) -> None:
            progress_events.append(("postfix", kwargs))

        def close(self) -> None:
            progress_events.append(("close", self.n))

    def fake_run_task_request(request, *, llm_base_url: str):
        from backend.models import StrategySummary, TaskRunResponse

        bar = service._create_request_progress_bar(request, position=0)
        try:
            progress = {
                "event": "step_completed",
                "step": 1,
                "action": "SELECT_CHILD_0",
                "compile_success": None,
                "visible_test_passed": None,
                "label_tier": "dsl_only",
            }
            bar.update(progress["step"])
            bar.set_postfix(
                action=progress["action"],
                compile=progress["compile_success"],
                visible=progress["visible_test_passed"],
                tier=progress["label_tier"],
                refresh=False,
            )
            strategy = StrategySummary(
                policy="heuristic",
                task_id="plugin_task",
                family="custom_single_file",
                total_reward=1.0,
                steps=1,
                compile_successes=0,
                visible_passes=0,
            )
            return TaskRunResponse(
                strategy=strategy,
                root_candidates=[],
                nodes=[],
                edges=[],
                transitions=[],
            )
        finally:
            bar.close()

    monkeypatch.setattr(service, "tqdm", FakeTqdm)
    monkeypatch.setattr(service, "run_task_request", fake_run_task_request)

    app = create_app("http://127.0.0.1:8080")
    client = TestClient(app)
    response = client.post(
        "/api/task/run",
        json={
            "prompt": "fix this",
            "files": {"foo.py": "print('x')\n"},
            "target_files": ["foo.py"],
            "policy": "heuristic",
            "max_steps": 4,
        },
    )
    assert response.status_code == 200
    assert ("update", 1) in progress_events
    assert any(event[0] == "postfix" for event in progress_events)
    assert any(event[0] == "close" for event in progress_events)


def test_request_progress_reporter_tracks_llm_and_verifier_events(monkeypatch) -> None:
    from backend import service
    from backend.models import TaskRunRequest

    progress_events: list[tuple[str, object]] = []

    class FakeTqdm:
        def __init__(
            self,
            total: int,
            desc: str,
            unit: str,
            leave: bool,
            position: int = 0,
            dynamic_ncols: bool = False,
        ) -> None:
            self.total = total
            self.desc = desc
            self.unit = unit
            self.leave = leave
            self.position = position
            self.dynamic_ncols = dynamic_ncols
            self.n = 0

        def update(self, amount: int) -> None:
            self.n += amount
            progress_events.append(("update", amount))

        def set_postfix(self, **kwargs) -> None:
            progress_events.append(("postfix", kwargs))

        def close(self) -> None:
            progress_events.append(("close", self.n))

    monkeypatch.setattr(service, "tqdm", FakeTqdm)

    request = TaskRunRequest(
        prompt="fix this",
        files={"foo.py": "print('x')\n"},
        target_files=["foo.py"],
        policy="heuristic",
        max_steps=4,
    )
    reporter = service._RequestProgressReporter(request)
    try:
        reporter.on_progress({"event": "search_started", "max_steps": 4})
        reporter.on_progress({"event": "llm_request_started", "request_label": "proposal", "mode": "chat", "attempt": 1})
        reporter.on_progress(
            {"event": "llm_request_completed", "request_label": "proposal", "mode": "chat", "attempt": 1, "elapsed_s": 1.25}
        )
        reporter.on_progress({"event": "verification_started", "plan_id": "plan_001"})
        reporter.on_progress({"event": "compile_completed", "compile_success": True, "elapsed_s": 2.0})
        reporter.on_progress({"event": "tests_completed", "visible_test_passed": True, "elapsed_s": 0.5})
        reporter.on_progress(
            {
                "event": "step_completed",
                "step": 1,
                "action": "SELECT_CHILD_0",
                "compile_success": True,
                "visible_test_passed": True,
                "label_tier": "visible_test",
            }
        )
    finally:
        reporter.close(status="completed")

    postfixes = [payload for kind, payload in progress_events if kind == "postfix"]
    assert ("update", 1) in progress_events
    assert any(payload.get("phase") == "proposal" for payload in postfixes)
    assert any(payload.get("llm_t") == "1.2s" for payload in postfixes)
    assert any(payload.get("compile_t") == "2.0s" for payload in postfixes)
    assert any(payload.get("test_t") == "0.5s" for payload in postfixes)
    assert any(event == ("close", 1) for event in progress_events)


def test_run_async_endpoints(monkeypatch) -> None:
    from backend import api
    from backend.models import JobProgressSnapshot, StrategySummary, TaskRunJobStatusResponse, TaskRunResponse

    def fake_start_task_run(self, request, *, llm_base_url: str):
        from backend.models import AsyncJobAcceptedResponse

        return AsyncJobAcceptedResponse(job_id="job-run-1", kind="task_run")

    def fake_get_task_run(self, job_id: str, *, graph_event_cursor: int = 0):
        strategy = StrategySummary(
            policy="heuristic",
            task_id="plugin_task",
            family="custom_single_file",
            total_reward=1.0,
            steps=2,
            compile_successes=1,
            visible_passes=1,
        )
        return TaskRunJobStatusResponse(
            job_id=job_id,
            status="completed",
            progress=JobProgressSnapshot(phase="done", current_step=2, max_steps=2, elapsed_s=1.2),
            result=TaskRunResponse(
                strategy=strategy,
                root_candidates=[],
                nodes=[],
                edges=[],
                transitions=[],
                search_graph_events=[{"type": "search_started", "run_id": "heuristic"}],
            ),
        )

    monkeypatch.setattr(api.AsyncJobStore, "start_task_run", fake_start_task_run)
    monkeypatch.setattr(api.AsyncJobStore, "get_task_run", fake_get_task_run)

    app = create_app("http://127.0.0.1:8080")
    client = TestClient(app)
    accepted = client.post(
        "/api/task/run_async",
        json={
            "prompt": "fix this",
            "files": {"foo.py": "print('x')\n"},
            "target_files": ["foo.py"],
            "policy": "heuristic",
        },
    )
    assert accepted.status_code == 200
    assert accepted.json()["job_id"] == "job-run-1"

    status = client.get("/api/task/run_async/job-run-1")
    assert status.status_code == 200
    payload = status.json()
    assert payload["status"] == "completed"
    assert payload["progress"]["phase"] == "done"
    assert payload["result"]["strategy"]["policy"] == "heuristic"
    assert payload["result"]["search_graph_events"][0]["type"] == "search_started"


def test_run_async_status_ignores_legacy_graph_cursor_query(monkeypatch) -> None:
    from backend import api
    from backend.models import JobProgressSnapshot, TaskRunJobStatusResponse

    def fake_start_task_run(self, request, *, llm_base_url: str):
        from backend.models import AsyncJobAcceptedResponse

        return AsyncJobAcceptedResponse(job_id="job-run-graph-1", kind="task_run")

    def fake_get_task_run(self, job_id: str):
        return TaskRunJobStatusResponse(
            job_id=job_id,
            status="running",
            progress=JobProgressSnapshot(phase="search", current_step=1, max_steps=8, elapsed_s=0.5),
        )

    monkeypatch.setattr(api.AsyncJobStore, "start_task_run", fake_start_task_run)
    monkeypatch.setattr(api.AsyncJobStore, "get_task_run", fake_get_task_run)

    app = create_app("http://127.0.0.1:8080")
    client = TestClient(app)
    status = client.get("/api/task/run_async/job-run-graph-1?graph_event_cursor=7")
    assert status.status_code == 200
    payload = status.json()
    assert payload["status"] == "running"
    assert "graph_event_cursor" not in payload
    assert "graph_events" not in payload


def test_compare_async_endpoints(monkeypatch) -> None:
    from backend import api
    from backend.models import CompareJobStatusResponse, CompareStrategiesResponse, JobProgressSnapshot, StrategySummary, TaskRunResponse

    def fake_start_compare(self, request, *, llm_base_url: str):
        from backend.models import AsyncJobAcceptedResponse

        return AsyncJobAcceptedResponse(job_id="job-compare-1", kind="compare")

    def fake_get_compare(self, job_id: str, *, graph_event_cursor: int = 0):
        strategy = StrategySummary(
            policy="heuristic",
            task_id="plugin_task",
            family="custom_single_file",
            total_reward=1.0,
            steps=2,
            compile_successes=1,
            visible_passes=1,
        )
        return CompareJobStatusResponse(
            job_id=job_id,
            status="completed",
            progress=JobProgressSnapshot(
                phase="done",
                policy="rainbow",
                current_policy_index=2,
                total_policies=2,
                current_step=8,
                max_steps=8,
                elapsed_s=2.4,
            ),
            result=CompareStrategiesResponse(
                strategies=[
                    TaskRunResponse(
                        strategy=strategy,
                        root_candidates=[],
                        nodes=[],
                        edges=[],
                        transitions=[],
                    )
                ]
            ),
        )

    monkeypatch.setattr(api.AsyncJobStore, "start_compare", fake_start_compare)
    monkeypatch.setattr(api.AsyncJobStore, "get_compare", fake_get_compare)

    app = create_app("http://127.0.0.1:8080")
    client = TestClient(app)
    accepted = client.post(
        "/api/task/compare_async",
        json={
            "prompt": "fix this",
            "files": {"foo.py": "print('x')\n"},
            "target_files": ["foo.py"],
            "policies": ["heuristic", "rainbow"],
        },
    )
    assert accepted.status_code == 200
    assert accepted.json()["job_id"] == "job-compare-1"

    status = client.get("/api/task/compare_async/job-compare-1")
    assert status.status_code == 200
    payload = status.json()
    assert payload["status"] == "completed"
    assert payload["progress"]["current_policy_index"] == 2
    assert payload["result"]["strategies"][0]["strategy"]["policy"] == "heuristic"
