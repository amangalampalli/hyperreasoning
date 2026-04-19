from __future__ import annotations

import gzip
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from backend.config import BackendConfig, SupabaseConfig, load_backend_config
from backend.models import (
    ClientContext,
    CompareStrategiesRequest,
    CompareStrategiesResponse,
    StrategySummary,
    TaskRunRequest,
    TaskRunResponse,
)
from backend.run_cache import (
    RunCache,
    SupabaseRunBackup,
    compute_comparison_cache_key,
    compute_run_cache_key,
    load_cached_run,
)


def _disabled_config() -> BackendConfig:
    return BackendConfig(supabase=SupabaseConfig(url=None, key=None, bucket=None))


def _context(tmp_path: Path) -> ClientContext:
    return ClientContext(
        project_id="project-test",
        project_name="Project Test",
        project_root=str(tmp_path),
        task_root=str(tmp_path / "task"),
        active_file=str(tmp_path / "task" / "foo.py"),
    )


def _request(tmp_path: Path, *, prompt: str = "fix this") -> TaskRunRequest:
    return TaskRunRequest(
        client_context=_context(tmp_path),
        prompt=prompt,
        files={"foo.py": "print('x')\n"},
        target_files=["foo.py"],
        policy="heuristic",
    )


def _response() -> TaskRunResponse:
    return TaskRunResponse(
        strategy=StrategySummary(
            policy="heuristic",
            task_id="plugin_task",
            family="custom_single_file",
            total_reward=1.0,
            steps=1,
            compile_successes=1,
            visible_passes=1,
            elapsed_s=0.1,
            llm_requests=0,
        ),
        root_candidates=[],
        nodes=[],
        edges=[],
        transitions=[],
        search_graph_events=[{"type": "search_started", "run_id": "heuristic"}],
    )


def _compare_request(tmp_path: Path) -> CompareStrategiesRequest:
    return CompareStrategiesRequest(
        client_context=_context(tmp_path),
        prompt="compare these",
        files={"foo.py": "print('x')\n"},
        target_files=["foo.py"],
        policies=["heuristic", "oneshot"],
    )


def _compare_response() -> CompareStrategiesResponse:
    return CompareStrategiesResponse(
        strategies=[
            _response(),
            _response().model_copy(
                update={
                    "strategy": _response().strategy.model_copy(
                        update={"policy": "oneshot", "total_reward": 0.5, "visible_passes": 0}
                    )
                }
            ),
        ]
    )


def test_dotenv_config_loading(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("HYPERREASONING_SUPABASE_URL", raising=False)
    monkeypatch.delenv("HYPERREASONING_SUPABASE_KEY", raising=False)
    monkeypatch.delenv("HYPERREASONING_SUPABASE_BUCKET", raising=False)
    monkeypatch.delenv("HYPERREASONING_SUPABASE_RUNS_TABLE", raising=False)
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "HYPERREASONING_SUPABASE_URL=https://example.supabase.co",
                "HYPERREASONING_SUPABASE_KEY=secret-value",
                "HYPERREASONING_SUPABASE_BUCKET=runs",
                "HYPERREASONING_SUPABASE_RUNS_TABLE=custom_runs",
            ]
        ),
        encoding="utf-8",
    )

    config = load_backend_config(env_path)

    assert config.supabase.enabled is True
    assert config.supabase.url == "https://example.supabase.co"
    assert config.supabase.key == "secret-value"
    assert config.supabase.bucket == "runs"
    assert config.supabase.runs_table == "custom_runs"


def test_missing_dotenv_disables_cloud(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("HYPERREASONING_SUPABASE_URL", raising=False)
    monkeypatch.delenv("HYPERREASONING_SUPABASE_KEY", raising=False)
    monkeypatch.delenv("HYPERREASONING_SUPABASE_BUCKET", raising=False)
    monkeypatch.delenv("HYPERREASONING_SUPABASE_RUNS_TABLE", raising=False)

    config = load_backend_config(tmp_path / ".env")

    assert config.supabase.enabled is False


def test_cache_key_uses_executable_request_not_project_identity(tmp_path: Path) -> None:
    first = _request(tmp_path)
    second = first.model_copy(
        update={
            "client_context": ClientContext(
                project_id="other",
                project_name="Other",
                project_root=str(tmp_path / "other"),
            )
        }
    )
    changed = first.model_copy(update={"files": {"foo.py": "print('y')\n"}})

    assert compute_run_cache_key(first) == compute_run_cache_key(second)
    assert compute_run_cache_key(first) != compute_run_cache_key(changed)


def test_comparison_cache_key_uses_policy_set(tmp_path: Path) -> None:
    first = _compare_request(tmp_path)
    same_different_project = first.model_copy(
        update={
            "client_context": ClientContext(
                project_id="other",
                project_name="Other",
                project_root=str(tmp_path / "other"),
            )
        }
    )
    changed = first.model_copy(update={"policies": ["heuristic", "rainbow"]})

    assert compute_comparison_cache_key(first) == compute_comparison_cache_key(same_different_project)
    assert compute_comparison_cache_key(first) != compute_comparison_cache_key(changed)


def test_local_cache_stores_lists_and_loads_package(tmp_path: Path) -> None:
    cache = RunCache.from_context(_context(tmp_path))
    request = _request(tmp_path)
    response = _response()

    item = cache.store_task_run(request=request, response=response, backend_config=_disabled_config())
    listed = cache.list_runs(limit=10, offset=0)
    loaded = cache.load_run(item.run_id)

    assert (tmp_path / ".hyper" / "runs.sqlite3").exists()
    assert listed[0].run_id == item.run_id
    assert loaded is not None
    assert loaded.result.cache_hit is True
    assert loaded.result.strategy.visible_passes == 1
    assert loaded.request.prompt == request.prompt


def test_local_cache_stores_comparison_as_one_history_item(tmp_path: Path) -> None:
    cache = RunCache.from_context(_context(tmp_path))
    request = _compare_request(tmp_path)
    response = _compare_response()

    item = cache.store_comparison_run(request=request, response=response, backend_config=_disabled_config())
    listed = cache.list_runs(limit=10, offset=0)
    loaded = cache.load_run(item.run_id)

    assert listed == [item]
    assert item.policy == "comparison"
    assert item.visible_passes == 1
    assert loaded is not None
    assert loaded.kind == "comparison"
    assert loaded.result is None
    assert loaded.compare_request is not None
    assert loaded.compare_result is not None
    assert loaded.compare_result.cache_hit is True
    assert len(loaded.compare_result.strategies) == 2


def test_cached_task_run_short_circuits_computation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from backend import service

    request = _request(tmp_path)
    cache = RunCache.from_context(_context(tmp_path))
    cache.store_task_run(request=request, response=_response(), backend_config=_disabled_config())

    def fail_search(*args, **kwargs):
        raise AssertionError("search should not run on cache hit")

    monkeypatch.setattr(service, "run_single_task_search", fail_search)

    response = service.run_task_request(request, llm_base_url="http://127.0.0.1:8080", backend_config=_disabled_config())

    assert response.cache_hit is True
    assert response.strategy.visible_passes == 1


def test_task_run_stores_with_backend_config_not_search_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from backend import service

    request = _request(tmp_path)
    config = BackendConfig(supabase=SupabaseConfig(url="https://example.supabase.co", key="secret", bucket="runs"))
    uploaded: list[str] = []

    def fake_search(*args, **kwargs):
        class FakeResult:
            policy = "heuristic"
            task_id = "plugin_task"
            family = "custom_single_file"
            total_reward = 1.0
            steps = 1
            compile_successes = 1
            visible_passes = 1
            verifier_summary = {}
            best_bank_id = "root"
            best_verification = {"compile_success": True, "visible_test_passed": True}
            root_candidates = []
            plan_bank = {}
            best_plan = None
            best_compiled_files = {}

            class Episode:
                nodes = []
                edges = []
                transitions = []

            episode = Episode()

        return FakeResult()

    class FakeBackup:
        enabled = True

        def __init__(self, supabase_config):
            assert supabase_config.bucket == "runs"

        def upload_run(self, *, item, package_bytes):
            uploaded.append(item.run_id)
            return item.model_copy(update={"cloud_status": "synced", "cloud_object_path": f"objects/{item.run_id}.json.gz"})

    monkeypatch.setattr(service, "run_single_task_search", fake_search)
    monkeypatch.setattr("backend.run_cache.SupabaseRunBackup", FakeBackup)

    response = service.run_task_request(request, llm_base_url="http://127.0.0.1:8080", backend_config=config)

    assert response.cloud_status == "synced"
    assert uploaded == [response.run_id]


def test_task_run_response_includes_terminal_diagnostics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from backend import service

    request = _request(tmp_path)

    class FakeTransition:
        action = "COMPILE_TO_CODE"
        state = {"current_bank_id": "bank_00001"}
        info = {
            "compile_success": True,
            "visible_test_passed": False,
            "visible_test_returncode": 1,
            "visible_test_stdout": "visible stdout",
            "visible_test_stderr": "visible stderr",
            "hidden_test_passed": False,
            "hidden_test_returncode": 2,
            "hidden_test_stdout": "hidden stdout",
            "hidden_test_stderr": "hidden stderr",
            "compile_error": None,
            "label_tier": "visible_test",
        }

        def model_dump(self):
            return {"action": self.action, "state": self.state, "info": self.info}

    def fake_search(*args, **kwargs):
        return SimpleNamespace(
            policy="heuristic",
            task_id="plugin_task",
            family="custom_single_file",
            total_reward=0.5,
            steps=1,
            compile_successes=1,
            visible_passes=0,
            verifier_summary={},
            best_bank_id="bank_00001",
            best_verification={
                "compile_success": True,
                "visible_test_passed": False,
                "visible_test_returncode": 1,
                "visible_test_stdout": "visible stdout",
                "visible_test_stderr": "visible stderr",
                "hidden_test_passed": False,
                "hidden_test_returncode": 2,
                "hidden_test_stdout": "hidden stdout",
                "hidden_test_stderr": "hidden stderr",
            },
            root_candidates=[],
            plan_bank={
                "entries": {
                    "bank_00001": {
                        "plan_signature": "sig-1",
                        "plan": {
                            "plan_id": "plan-1",
                            "strategy": "patch failure",
                            "target_files": ["foo.py"],
                        },
                    }
                }
            },
            best_plan=None,
            best_compiled_files={},
            episode=SimpleNamespace(nodes=[], edges=[], transitions=[FakeTransition()]),
        )

    monkeypatch.setattr(service, "run_single_task_search", fake_search)

    response = service.run_task_request(
        request,
        llm_base_url="http://127.0.0.1:8080",
        backend_config=_disabled_config(),
        store_result=False,
    )

    assert len(response.diagnostics) == 1
    diagnostic = response.diagnostics[0]
    assert diagnostic.is_best is True
    assert diagnostic.status == "hidden_failed"
    assert diagnostic.plan_id == "plan-1"
    assert diagnostic.strategy == "patch failure"
    assert diagnostic.target_files == ["foo.py"]
    assert diagnostic.visible_test_stderr == "visible stderr"
    assert diagnostic.hidden_test_stdout == "hidden stdout"


def test_task_run_summary_reports_exact_test_counts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from backend import service

    request = _request(tmp_path)

    def fake_search(*args, **kwargs):
        return SimpleNamespace(
            policy="heuristic",
            task_id="plugin_task",
            family="custom_single_file",
            total_reward=0.5,
            steps=1,
            compile_successes=1,
            visible_passes=0,
            verifier_summary={"hidden_test_passes": 1},
            best_bank_id="bank_00001",
            best_verification={
                "compile_success": True,
                "visible_test_passed": False,
                "visible_test_stdout": "test_ok ... ok\ntest_bad ... FAIL\n\nRan 2 tests in 0.001s\n\nFAILED (failures=1)\n",
                "visible_test_stderr": "",
                "hidden_test_passed": True,
                "hidden_test_stdout": "..\n----------------------------------------------------------------------\nRan 2 tests in 0.000s\n\nOK\n",
                "hidden_test_stderr": "",
            },
            root_candidates=[],
            plan_bank={"entries": {}},
            best_plan=None,
            best_compiled_files={},
            episode=SimpleNamespace(nodes=[], edges=[], transitions=[]),
        )

    monkeypatch.setattr(service, "run_single_task_search", fake_search)

    response = service.run_task_request(
        request,
        llm_base_url="http://127.0.0.1:8080",
        backend_config=_disabled_config(),
        store_result=False,
    )

    assert response.strategy.visible_tests_passed == 1
    assert response.strategy.visible_tests_total == 2
    assert response.strategy.hidden_tests_passed == 2
    assert response.strategy.hidden_tests_total == 2
    assert response.strategy.tests_passed == 3
    assert response.strategy.tests_total == 4
    assert response.strategy.fraction_tests_passed == 0.75


def test_compile_failure_diagnostics_are_emitted_from_transitions() -> None:
    from backend import service

    result = SimpleNamespace(
        policy="heuristic",
        best_bank_id="bank_00002",
        best_verification=None,
        plan_bank={
            "entries": {
                "bank_00002": {
                    "plan_signature": "sig-2",
                    "plan": {
                        "plan_id": "plan-2",
                        "strategy": "syntax repair",
                        "target_files": ["foo.py"],
                    },
                }
            }
        },
        episode=SimpleNamespace(
            transitions=[
                SimpleNamespace(
                    action="COMPILE_TO_CODE",
                    state={"current_bank_id": "bank_00002"},
                    info={"compile_success": False, "compile_error": "foo.py:1: invalid syntax"},
                )
            ]
        ),
    )

    diagnostics = service._build_diagnostics(result)

    assert len(diagnostics) == 1
    assert diagnostics[0].status == "compile_failed"
    assert diagnostics[0].compile_error == "foo.py:1: invalid syntax"
    assert diagnostics[0].is_best is True


def test_compare_strategies_stores_grouped_comparison_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from backend import service

    request = _compare_request(tmp_path)

    def fake_search(task, config, *, policy, **kwargs):
        base = _response()
        visible_passes = 1 if policy == "heuristic" else 0
        return SimpleNamespace(
            policy=policy,
            task_id="plugin_task",
            family="custom_single_file",
            total_reward=1.0 if policy == "heuristic" else 0.5,
            steps=1,
            compile_successes=1,
            visible_passes=visible_passes,
            verifier_summary={},
            best_bank_id="root",
            best_verification={"compile_success": True, "visible_test_passed": visible_passes > 0},
            root_candidates=[],
            plan_bank={},
            best_plan=None,
            best_compiled_files={},
            episode=SimpleNamespace(nodes=base.nodes, edges=base.edges, transitions=[]),
        )

    monkeypatch.setattr(service, "run_single_task_search", fake_search)

    response = service.compare_strategies(
        request,
        llm_base_url="http://127.0.0.1:8080",
        backend_config=_disabled_config(),
    )
    cache = RunCache.from_context(_context(tmp_path))
    listed = cache.list_runs(limit=10, offset=0)
    loaded = cache.find_by_cache_key(response.cache_key or "", policy="comparison")
    second = service.compare_strategies(
        request,
        llm_base_url="http://127.0.0.1:8080",
        backend_config=_disabled_config(),
    )

    assert response.run_id is not None
    assert [item.policy for item in listed] == ["comparison"]
    assert loaded is not None
    assert loaded.compare_result is not None
    assert second.cache_hit is True
    assert second.run_id == response.run_id


def test_supabase_backup_upload_and_download(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cache = RunCache.from_context(_context(tmp_path))
    item = cache.store_task_run(request=_request(tmp_path), response=_response(), backend_config=_disabled_config())
    package_bytes = cache.package_bytes(item.run_id)
    uploads: dict[str, bytes] = {}
    upserts: list[dict] = []

    class FakeStorageBucket:
        def upload(self, *, path, file, file_options):
            uploads[path] = file

        def download(self, path):
            return uploads[path]

    class FakeStorage:
        def from_(self, bucket):
            assert bucket == "runs"
            return FakeStorageBucket()

    class FakeExecute:
        data = []

        def execute(self):
            return self

    class FakeTable:
        def upsert(self, row, on_conflict):
            upserts.append(row)
            assert on_conflict == "run_id"
            return FakeExecute()

    class FakeClient:
        storage = FakeStorage()

        def table(self, name):
            assert name == "hyperreasoning_runs"
            return FakeTable()

    backup = SupabaseRunBackup(SupabaseConfig(url="https://example.supabase.co", key="secret", bucket="runs"))
    monkeypatch.setattr(backup, "_supabase", lambda: FakeClient())

    uploaded = backup.upload_run(item=item, package_bytes=package_bytes)
    downloaded, downloaded_item = backup.download_run(project_id=item.project_id, run_id=item.run_id, item=uploaded)

    assert uploaded.cloud_status == "synced"
    assert uploaded.cloud_object_path in uploads
    assert upserts[0]["run_id"] == item.run_id
    assert downloaded == package_bytes
    assert downloaded_item.run_id == item.run_id


def test_restore_rejects_checksum_mismatch(tmp_path: Path) -> None:
    cache = RunCache.from_context(_context(tmp_path))
    item = cache.store_task_run(request=_request(tmp_path), response=_response(), backend_config=_disabled_config())
    package_bytes = cache.package_bytes(item.run_id)

    with pytest.raises(ValueError, match="checksum"):
        cache.restore_package(gzip.compress(b'{"kind":"broken"}'), expected_item=item)

    with pytest.raises(ValueError, match="checksum"):
        cache.restore_package(package_bytes + b"changed", expected_item=item)


def test_run_history_api_lists_and_loads_local_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from backend import api

    monkeypatch.setattr(api, "load_backend_config", lambda: _disabled_config())
    cache = RunCache.from_context(_context(tmp_path))
    item = cache.store_task_run(request=_request(tmp_path), response=_response(), backend_config=_disabled_config())
    client = TestClient(api.create_app("http://127.0.0.1:8080"))

    listed = client.get(
        "/api/runs",
        params={"project_id": "project-test", "project_root": str(tmp_path)},
    )
    loaded = client.get(
        f"/api/runs/{item.run_id}",
        params={"project_id": "project-test", "project_root": str(tmp_path)},
    )
    synced = client.post(
        "/api/runs/sync",
        json={
            "client_context": {
                "project_id": "project-test",
                "project_root": str(tmp_path),
            }
        },
    )

    assert listed.status_code == 200
    assert listed.json()["items"][0]["run_id"] == item.run_id
    assert loaded.status_code == 200
    assert loaded.json()["result"]["strategy"]["visible_passes"] == 1
    assert synced.status_code == 200
    assert synced.json()["cloud_enabled"] is False


def test_load_cached_run_restores_from_cloud_after_local_delete(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    restore_root = tmp_path / "restore"
    source = RunCache.from_context(_context(source_root))
    item = source.store_task_run(request=_request(source_root), response=_response(), backend_config=_disabled_config())
    package_bytes = source.package_bytes(item.run_id)
    cloud_item = item.model_copy(update={"local_status": "missing", "cloud_status": "synced"})

    class FakeBackup:
        enabled = True

        def __init__(self, config):
            pass

        def download_run(self, *, project_id, run_id, item=None):
            assert run_id == cloud_item.run_id
            return package_bytes, cloud_item

    monkeypatch.setattr("backend.run_cache.SupabaseRunBackup", FakeBackup)

    restored = load_cached_run(
        item.run_id,
        context=_context(restore_root),
        backend_config=BackendConfig(supabase=SupabaseConfig(url="https://example.supabase.co", key="secret", bucket="runs")),
    )

    assert restored.result.strategy.visible_passes == 1
    assert (restore_root / ".hyper" / "runs" / f"{item.run_id}.json.gz").exists()
