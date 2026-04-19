"""Local and Supabase-backed cache for plugin task runs."""

from __future__ import annotations

import gzip
import hashlib
import logging
from pathlib import Path
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import orjson

from backend.config import BackendConfig, SupabaseConfig, load_backend_config
from backend.models import (
    ClientContext,
    CompareStrategiesRequest,
    CompareStrategiesResponse,
    RunHistoryItem,
    RunHistoryResponse,
    RunLoadResponse,
    RunSyncRequest,
    RunSyncResponse,
    TaskRunRequest,
    TaskRunResponse,
)


LOGGER = logging.getLogger("backend.run_cache")
PACKAGE_KIND = "hyperreasoning_run_package_v1"
PACKAGE_SCHEMA_VERSION = 1
SQLITE_NAME = "runs.sqlite3"
PACKAGES_DIR = "runs"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def canonical_json_bytes(value: Any) -> bytes:
    return orjson.dumps(value, option=orjson.OPT_SORT_KEYS)


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def compute_run_cache_key(request: TaskRunRequest) -> str:
    """Compute a stable cache key for a request's executable inputs."""

    payload = request.model_dump(mode="json", exclude={"client_context"})
    payload["checkpoint_fingerprint"] = _checkpoint_fingerprint(request.checkpoint_path)
    return sha256_hex(canonical_json_bytes(payload))


def compute_comparison_cache_key(request: CompareStrategiesRequest) -> str:
    """Compute a stable cache key for a full strategy comparison."""

    payload = request.model_dump(mode="json", exclude={"client_context", "policy"})
    payload["checkpoint_fingerprint"] = _checkpoint_fingerprint(request.checkpoint_path)
    return sha256_hex(canonical_json_bytes(payload))


def cache_for_context(context: ClientContext | None) -> "RunCache | None":
    if context is None or not context.project_root:
        return None
    return RunCache.from_context(context)


def list_run_history(
    context: ClientContext,
    *,
    limit: int,
    offset: int,
    query: str | None = None,
    backend_config: BackendConfig | None = None,
) -> RunHistoryResponse:
    config = backend_config or load_backend_config()
    backup = SupabaseRunBackup(config.supabase)
    cache = cache_for_context(context)
    errors: list[str] = []
    local_items = cache.list_runs(limit=limit, offset=offset, query=query) if cache else []
    by_id = {item.run_id: item for item in local_items}

    project_id = _project_id(context)
    if backup.enabled:
        try:
            for cloud_item in backup.list_runs(project_id=project_id, limit=limit, offset=offset, query=query):
                if cloud_item.run_id not in by_id:
                    by_id[cloud_item.run_id] = cloud_item.model_copy(update={"local_status": "missing"})
        except Exception as exc:
            errors.append(_safe_error(exc))

    items = sorted(by_id.values(), key=lambda item: item.created_at, reverse=True)
    return RunHistoryResponse(items=items[:limit], cloud_enabled=backup.enabled, errors=errors)


def load_cached_run(
    run_id: str,
    *,
    context: ClientContext | None,
    backend_config: BackendConfig | None = None,
) -> RunLoadResponse:
    config = backend_config or load_backend_config()
    backup = SupabaseRunBackup(config.supabase)
    cache = cache_for_context(context)
    if cache is not None:
        loaded = cache.load_run(run_id)
        if loaded is not None:
            return loaded

    if context is None:
        raise KeyError(run_id)
    if not backup.enabled:
        raise KeyError(run_id)

    package_bytes, cloud_item = backup.download_run(project_id=_project_id(context), run_id=run_id)
    if cache is None:
        cache = RunCache.from_context(context)
    return cache.restore_package(package_bytes, expected_item=cloud_item)


def sync_runs(request: RunSyncRequest, *, backend_config: BackendConfig | None = None) -> RunSyncResponse:
    config = backend_config or load_backend_config()
    backup = SupabaseRunBackup(config.supabase)
    if not backup.enabled:
        return RunSyncResponse(cloud_enabled=False)

    cache = RunCache.from_context(request.client_context)
    uploaded = 0
    downloaded = 0
    failed = 0
    errors: list[str] = []

    for item in cache.list_runs(limit=request.limit, offset=0, include_failed=True):
        if item.cloud_status not in {"pending", "failed"}:
            continue
        try:
            package_bytes = cache.package_bytes(item.run_id)
            uploaded_item = backup.upload_run(item=item, package_bytes=package_bytes)
            cache.update_cloud_status(
                item.run_id,
                cloud_status=uploaded_item.cloud_status,
                cloud_object_path=uploaded_item.cloud_object_path,
            )
            uploaded += 1
        except Exception as exc:
            failed += 1
            errors.append(_safe_error(exc))
            cache.update_cloud_status(item.run_id, cloud_status="failed")

    try:
        for cloud_item in backup.list_runs(project_id=_project_id(request.client_context), limit=request.limit, offset=0):
            if cache.has_package(cloud_item.run_id):
                continue
            try:
                package_bytes, downloaded_item = backup.download_run(
                    project_id=cloud_item.project_id,
                    run_id=cloud_item.run_id,
                    item=cloud_item,
                )
                cache.restore_package(package_bytes, expected_item=downloaded_item)
                downloaded += 1
            except Exception as exc:
                failed += 1
                errors.append(_safe_error(exc))
    except Exception as exc:
        failed += 1
        errors.append(_safe_error(exc))

    return RunSyncResponse(
        cloud_enabled=True,
        uploaded=uploaded,
        downloaded=downloaded,
        failed=failed,
        errors=errors,
    )


class RunCache:
    def __init__(self, *, project_root: Path, project_id: str, project_name: str | None = None) -> None:
        self.project_root = project_root
        self.project_id = project_id
        self.project_name = project_name
        self.cache_root = project_root / ".hyper"
        self.packages_root = self.cache_root / PACKAGES_DIR
        self.db_path = self.cache_root / SQLITE_NAME
        self._lock = threading.Lock()
        self._ensure_schema()

    @classmethod
    def from_context(cls, context: ClientContext) -> "RunCache":
        if not context.project_root:
            raise ValueError("client_context.project_root is required for run cache")
        return cls(
            project_root=Path(context.project_root).expanduser().resolve(),
            project_id=_project_id(context),
            project_name=context.project_name,
        )

    def find_by_cache_key(self, cache_key: str, *, policy: str = "task_run") -> RunLoadResponse | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM runs WHERE cache_key = ? AND policy = ? ORDER BY created_at DESC LIMIT 1",
                (cache_key, policy),
            ).fetchone()
        if row is None:
            return None
        return self.load_run(str(row["run_id"]))

    def find_task_by_cache_key(self, cache_key: str) -> RunLoadResponse | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM runs WHERE cache_key = ? AND policy != 'comparison' ORDER BY created_at DESC LIMIT 1",
                (cache_key,),
            ).fetchone()
        if row is None:
            return None
        return self.load_run(str(row["run_id"]))

    def store_task_run(
        self,
        *,
        request: TaskRunRequest,
        response: TaskRunResponse,
        backend_config: BackendConfig | None = None,
    ) -> RunHistoryItem:
        config = backend_config or load_backend_config()
        backup = SupabaseRunBackup(config.supabase)
        run_id = response.run_id or uuid4().hex
        cache_key = response.cache_key or compute_run_cache_key(request)
        now = utc_now_iso()
        cloud_status = "pending" if backup.enabled else "disabled"
        response.run_id = run_id
        response.cache_key = cache_key
        response.cache_hit = False
        response.cloud_status = cloud_status

        package_payload = {
            "run_kind": "task_run",
            "created_at": now,
            "updated_at": now,
            "client_context": (request.client_context or ClientContext()).model_dump(mode="json"),
            "request": request.model_dump(mode="json"),
            "result": response.model_dump(mode="json"),
            "summary": _summary_payload(response),
        }
        package = _build_package(run_id=run_id, cache_key=cache_key, payload=package_payload)
        package_bytes = gzip.compress(canonical_json_bytes(package))
        package_sha = sha256_hex(package_bytes)
        item = _history_item_from_package(
            package=package,
            package_sha256=package_sha,
            project_id=self.project_id,
            project_name=self.project_name,
            project_root=str(self.project_root),
            cloud_status=cloud_status,
        )
        self._write_package_and_metadata(item=item, package_bytes=package_bytes)

        if backup.enabled:
            try:
                uploaded = backup.upload_run(item=item, package_bytes=package_bytes)
                item = uploaded
            except Exception as exc:
                LOGGER.warning("Supabase run backup failed for run_id=%s: %s", run_id, _safe_error(exc))
                item = item.model_copy(update={"cloud_status": "failed"})
            self.update_cloud_status(
                run_id,
                cloud_status=item.cloud_status,
                cloud_object_path=item.cloud_object_path,
            )
            response.cloud_status = item.cloud_status
        return item

    def store_comparison_run(
        self,
        *,
        request: CompareStrategiesRequest,
        response: CompareStrategiesResponse,
        backend_config: BackendConfig | None = None,
    ) -> RunHistoryItem:
        config = backend_config or load_backend_config()
        backup = SupabaseRunBackup(config.supabase)
        run_id = response.run_id or uuid4().hex
        cache_key = response.cache_key or compute_comparison_cache_key(request)
        now = utc_now_iso()
        cloud_status = "pending" if backup.enabled else "disabled"
        response.run_id = run_id
        response.cache_key = cache_key
        response.cache_hit = False
        response.cloud_status = cloud_status

        package_payload = {
            "run_kind": "comparison",
            "created_at": now,
            "updated_at": now,
            "client_context": (request.client_context or ClientContext()).model_dump(mode="json"),
            "request": request.model_dump(mode="json"),
            "comparison_result": response.model_dump(mode="json"),
            "summary": _comparison_summary_payload(response),
        }
        package = _build_package(run_id=run_id, cache_key=cache_key, payload=package_payload)
        package_bytes = gzip.compress(canonical_json_bytes(package))
        package_sha = sha256_hex(package_bytes)
        item = _history_item_from_package(
            package=package,
            package_sha256=package_sha,
            project_id=self.project_id,
            project_name=self.project_name,
            project_root=str(self.project_root),
            cloud_status=cloud_status,
        )
        self._write_package_and_metadata(item=item, package_bytes=package_bytes)

        if backup.enabled:
            try:
                uploaded = backup.upload_run(item=item, package_bytes=package_bytes)
                item = uploaded
            except Exception as exc:
                LOGGER.warning("Supabase comparison backup failed for run_id=%s: %s", run_id, _safe_error(exc))
                item = item.model_copy(update={"cloud_status": "failed"})
            self.update_cloud_status(
                run_id,
                cloud_status=item.cloud_status,
                cloud_object_path=item.cloud_object_path,
            )
            response.cloud_status = item.cloud_status
        return item

    def list_runs(
        self,
        *,
        limit: int,
        offset: int,
        query: str | None = None,
        include_failed: bool = False,
    ) -> list[RunHistoryItem]:
        where = ["project_id = ?"]
        params: list[Any] = [self.project_id]
        if query:
            where.append("(prompt_preview LIKE ? OR policy LIKE ? OR family LIKE ?)")
            pattern = f"%{query}%"
            params.extend([pattern, pattern, pattern])
        if not include_failed:
            where.append("local_status = 'available'")
        sql = (
            "SELECT * FROM runs WHERE "
            + " AND ".join(where)
            + " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        )
        params.extend([max(1, min(limit, 200)), max(0, offset)])
        with self._connect() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
        return [_item_from_row(row) for row in rows]

    def load_run(self, run_id: str) -> RunLoadResponse | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            return None
        item = _item_from_row(row)
        path = self._package_path(run_id)
        if not path.exists():
            return None
        package = _load_package_bytes(path.read_bytes())
        payload = package["payload"]
        if payload.get("run_kind") == "comparison":
            request = CompareStrategiesRequest.model_validate(payload["request"])
            result = CompareStrategiesResponse.model_validate(payload["comparison_result"])
            result.run_id = item.run_id
            result.cache_key = item.cache_key
            result.cache_hit = True
            result.cloud_status = item.cloud_status
            return RunLoadResponse(
                item=item,
                kind="comparison",
                compare_request=request,
                compare_result=result,
            )
        request = TaskRunRequest.model_validate(payload["request"])
        result = TaskRunResponse.model_validate(payload["result"])
        result.run_id = item.run_id
        result.cache_key = item.cache_key
        result.cache_hit = True
        result.cloud_status = item.cloud_status
        return RunLoadResponse(item=item, kind="task_run", request=request, result=result)

    def restore_package(self, package_bytes: bytes, *, expected_item: RunHistoryItem) -> RunLoadResponse:
        package_sha = sha256_hex(package_bytes)
        if expected_item.package_sha256 and package_sha != expected_item.package_sha256:
            raise ValueError("Downloaded run package checksum did not match metadata")
        package = _load_package_bytes(package_bytes)
        item = _history_item_from_package(
            package=package,
            package_sha256=package_sha,
            project_id=self.project_id,
            project_name=self.project_name,
            project_root=str(self.project_root),
            cloud_status=expected_item.cloud_status,
            cloud_object_path=expected_item.cloud_object_path,
        )
        self._write_package_and_metadata(item=item, package_bytes=package_bytes)
        loaded = self.load_run(item.run_id)
        if loaded is None:
            raise KeyError(item.run_id)
        return loaded

    def package_bytes(self, run_id: str) -> bytes:
        path = self._package_path(run_id)
        if not path.exists():
            raise KeyError(run_id)
        return path.read_bytes()

    def has_package(self, run_id: str) -> bool:
        return self._package_path(run_id).exists()

    def update_cloud_status(
        self,
        run_id: str,
        *,
        cloud_status: str,
        cloud_object_path: str | None = None,
    ) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    UPDATE runs
                    SET cloud_status = ?, cloud_object_path = COALESCE(?, cloud_object_path), updated_at = ?
                    WHERE run_id = ?
                    """,
                    (cloud_status, cloud_object_path, utc_now_iso(), run_id),
                )

    def _write_package_and_metadata(self, *, item: RunHistoryItem, package_bytes: bytes) -> None:
        self.packages_root.mkdir(parents=True, exist_ok=True)
        package_path = self._package_path(item.run_id)
        tmp_path = package_path.with_suffix(".tmp")
        tmp_path.write_bytes(package_bytes)
        tmp_path.replace(package_path)
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO runs (
                        run_id, cache_key, project_id, project_name, project_root, task_root, active_file,
                        prompt_preview, policy, family, created_at, updated_at, visible_passes,
                        hidden_passes, compile_successes, total_reward, elapsed_s, llm_requests,
                        local_status, cloud_status, cloud_object_path, package_sha256
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(run_id) DO UPDATE SET
                        cache_key = excluded.cache_key,
                        project_id = excluded.project_id,
                        project_name = excluded.project_name,
                        project_root = excluded.project_root,
                        task_root = excluded.task_root,
                        active_file = excluded.active_file,
                        prompt_preview = excluded.prompt_preview,
                        policy = excluded.policy,
                        family = excluded.family,
                        updated_at = excluded.updated_at,
                        visible_passes = excluded.visible_passes,
                        hidden_passes = excluded.hidden_passes,
                        compile_successes = excluded.compile_successes,
                        total_reward = excluded.total_reward,
                        elapsed_s = excluded.elapsed_s,
                        llm_requests = excluded.llm_requests,
                        local_status = excluded.local_status,
                        cloud_status = excluded.cloud_status,
                        cloud_object_path = excluded.cloud_object_path,
                        package_sha256 = excluded.package_sha256
                    """,
                    _item_sql_values(item),
                )

    def _ensure_schema(self) -> None:
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.packages_root.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    cache_key TEXT NOT NULL,
                    project_id TEXT NOT NULL,
                    project_name TEXT,
                    project_root TEXT,
                    task_root TEXT,
                    active_file TEXT,
                    prompt_preview TEXT NOT NULL,
                    policy TEXT NOT NULL,
                    family TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    visible_passes INTEGER NOT NULL DEFAULT 0,
                    hidden_passes INTEGER NOT NULL DEFAULT 0,
                    compile_successes INTEGER NOT NULL DEFAULT 0,
                    total_reward REAL NOT NULL DEFAULT 0,
                    elapsed_s REAL,
                    llm_requests INTEGER NOT NULL DEFAULT 0,
                    local_status TEXT NOT NULL DEFAULT 'available',
                    cloud_status TEXT NOT NULL DEFAULT 'unknown',
                    cloud_object_path TEXT,
                    package_sha256 TEXT
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_project_created ON runs(project_id, created_at DESC)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_cache_key ON runs(cache_key)")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _package_path(self, run_id: str) -> Path:
        return self.packages_root / f"{run_id}.json.gz"


class SupabaseRunBackup:
    def __init__(self, config: SupabaseConfig) -> None:
        self.config = config
        self._client: Any | None = None

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def upload_run(self, *, item: RunHistoryItem, package_bytes: bytes) -> RunHistoryItem:
        if not self.enabled:
            return item.model_copy(update={"cloud_status": "disabled"})
        object_path = self.object_path(item.project_id, item.run_id)
        client = self._supabase()
        client.storage.from_(self.config.bucket).upload(
            path=object_path,
            file=package_bytes,
            file_options={"content-type": "application/gzip", "cache-control": "3600", "upsert": "true"},
        )
        uploaded = item.model_copy(update={"cloud_status": "synced", "cloud_object_path": object_path})
        client.table(self.config.runs_table).upsert(
            _cloud_row(uploaded),
            on_conflict="run_id",
        ).execute()
        return uploaded

    def list_runs(
        self,
        *,
        project_id: str,
        limit: int,
        offset: int,
        query: str | None = None,
    ) -> list[RunHistoryItem]:
        if not self.enabled:
            return []
        query_builder = (
            self._supabase()
            .table(self.config.runs_table)
            .select("*")
            .eq("project_id", project_id)
            .order("created_at", desc=True)
            .range(max(0, offset), max(0, offset) + max(1, min(limit, 200)) - 1)
        )
        response = query_builder.execute()
        items = [_item_from_mapping(row, local_status="missing") for row in (response.data or [])]
        if query:
            needle = query.lower()
            items = [
                item
                for item in items
                if needle in item.prompt_preview.lower()
                or needle in item.policy.lower()
                or needle in item.family.lower()
            ]
        return items

    def download_run(
        self,
        *,
        project_id: str,
        run_id: str,
        item: RunHistoryItem | None = None,
    ) -> tuple[bytes, RunHistoryItem]:
        if not self.enabled:
            raise KeyError(run_id)
        resolved_item = item or self._get_item(project_id=project_id, run_id=run_id)
        object_path = resolved_item.cloud_object_path or self.object_path(project_id, run_id)
        data = self._supabase().storage.from_(self.config.bucket).download(object_path)
        if not isinstance(data, bytes):
            data = bytes(data)
        return data, resolved_item

    def object_path(self, project_id: str, run_id: str) -> str:
        return f"projects/{project_id}/runs/{run_id}.json.gz"

    def _get_item(self, *, project_id: str, run_id: str) -> RunHistoryItem:
        response = (
            self._supabase()
            .table(self.config.runs_table)
            .select("*")
            .eq("project_id", project_id)
            .eq("run_id", run_id)
            .execute()
        )
        rows = response.data or []
        if not rows:
            raise KeyError(run_id)
        return _item_from_mapping(rows[0], local_status="missing")

    def _supabase(self) -> Any:
        if not self.enabled:
            raise RuntimeError("Supabase is not configured")
        if self._client is None:
            from supabase import create_client

            self._client = create_client(self.config.url, self.config.key)
        return self._client


def _checkpoint_fingerprint(checkpoint_path: str | None) -> dict[str, Any] | None:
    if not checkpoint_path:
        return None
    path = Path(checkpoint_path).expanduser()
    try:
        stat = path.stat()
    except OSError:
        return {"path": str(path), "exists": False}
    return {
        "path": str(path),
        "exists": True,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _project_id(context: ClientContext) -> str:
    if context.project_id:
        return context.project_id
    basis = context.project_root or context.project_name or "hyperreasoning-project"
    return sha256_hex(basis.encode("utf-8"))[:24]


def _build_package(*, run_id: str, cache_key: str, payload: dict[str, Any]) -> dict[str, Any]:
    payload_sha = sha256_hex(canonical_json_bytes(payload))
    return {
        "kind": PACKAGE_KIND,
        "schema_version": PACKAGE_SCHEMA_VERSION,
        "run_id": run_id,
        "cache_key": cache_key,
        "payload_sha256": payload_sha,
        "payload": payload,
    }


def _load_package_bytes(package_bytes: bytes) -> dict[str, Any]:
    package = orjson.loads(gzip.decompress(package_bytes))
    if package.get("kind") != PACKAGE_KIND:
        raise ValueError("Unsupported run package kind")
    if int(package.get("schema_version") or 0) != PACKAGE_SCHEMA_VERSION:
        raise ValueError("Unsupported run package schema version")
    expected_sha = package.get("payload_sha256")
    actual_sha = sha256_hex(canonical_json_bytes(package.get("payload")))
    if expected_sha != actual_sha:
        raise ValueError("Run package payload checksum did not match")
    return package


def _history_item_from_package(
    *,
    package: dict[str, Any],
    package_sha256: str,
    project_id: str,
    project_name: str | None,
    project_root: str | None,
    cloud_status: str,
    cloud_object_path: str | None = None,
) -> RunHistoryItem:
    payload = package["payload"]
    run_kind = str(payload.get("run_kind") or "task_run")
    if run_kind == "comparison":
        request = CompareStrategiesRequest.model_validate(payload["request"])
        result = CompareStrategiesResponse.model_validate(payload["comparison_result"])
        summary = _comparison_summary_payload(result)
        policy = "comparison"
        visible_passes = int(summary["visible_passes"])
        hidden_passes = int(summary["hidden_passes"])
        compile_successes = int(summary["compile_successes"])
        total_reward = float(summary["total_reward"])
        elapsed_s = summary["elapsed_s"]
        llm_requests = int(summary["llm_requests"])
        family = request.family
    else:
        request = TaskRunRequest.model_validate(payload["request"])
        result = TaskRunResponse.model_validate(payload["result"])
        policy = result.strategy.policy
        family = result.strategy.family
        visible_passes = result.strategy.visible_passes
        hidden_passes = result.strategy.hidden_passes
        compile_successes = result.strategy.compile_successes
        total_reward = result.strategy.total_reward
        elapsed_s = result.strategy.elapsed_s
        llm_requests = result.strategy.llm_requests
    context = request.client_context or ClientContext()
    return RunHistoryItem(
        run_id=str(package["run_id"]),
        cache_key=str(package["cache_key"]),
        project_id=project_id,
        project_name=context.project_name or project_name,
        project_root=context.project_root or project_root,
        task_root=context.task_root,
        active_file=context.active_file,
        prompt_preview=_prompt_preview(request.prompt),
        policy=policy,
        family=family,
        created_at=str(payload.get("created_at") or utc_now_iso()),
        updated_at=str(payload.get("updated_at") or payload.get("created_at") or utc_now_iso()),
        visible_passes=visible_passes,
        hidden_passes=hidden_passes,
        compile_successes=compile_successes,
        total_reward=total_reward,
        elapsed_s=elapsed_s,
        llm_requests=llm_requests,
        local_status="available",
        cloud_status=cloud_status,  # type: ignore[arg-type]
        cloud_object_path=cloud_object_path,
        package_sha256=package_sha256,
    )


def _summary_payload(response: TaskRunResponse) -> dict[str, Any]:
    return {
        "policy": response.strategy.policy,
        "family": response.strategy.family,
        "visible_passes": response.strategy.visible_passes,
        "hidden_passes": response.strategy.hidden_passes,
        "compile_successes": response.strategy.compile_successes,
        "total_reward": response.strategy.total_reward,
        "elapsed_s": response.strategy.elapsed_s,
        "llm_requests": response.strategy.llm_requests,
    }


def _comparison_summary_payload(response: CompareStrategiesResponse) -> dict[str, Any]:
    strategies = response.strategies
    return {
        "policy": "comparison",
        "family": strategies[0].strategy.family if strategies else "custom_single_file",
        "visible_passes": max((item.strategy.visible_passes for item in strategies), default=0),
        "hidden_passes": max((item.strategy.hidden_passes for item in strategies), default=0),
        "compile_successes": max((item.strategy.compile_successes for item in strategies), default=0),
        "total_reward": max((item.strategy.total_reward for item in strategies), default=0.0),
        "elapsed_s": sum((item.strategy.elapsed_s or 0.0 for item in strategies), start=0.0) if strategies else None,
        "llm_requests": sum(item.strategy.llm_requests for item in strategies),
        "strategies": len(strategies),
    }


def _prompt_preview(prompt: str) -> str:
    compact = " ".join(prompt.split())
    return compact[:220] if compact else "(empty prompt)"


def _item_sql_values(item: RunHistoryItem) -> tuple[Any, ...]:
    return (
        item.run_id,
        item.cache_key,
        item.project_id,
        item.project_name,
        item.project_root,
        item.task_root,
        item.active_file,
        item.prompt_preview,
        item.policy,
        item.family,
        item.created_at,
        item.updated_at,
        item.visible_passes,
        item.hidden_passes,
        item.compile_successes,
        item.total_reward,
        item.elapsed_s,
        item.llm_requests,
        item.local_status,
        item.cloud_status,
        item.cloud_object_path,
        item.package_sha256,
    )


def _item_from_row(row: sqlite3.Row) -> RunHistoryItem:
    return _item_from_mapping(dict(row), local_status=str(row["local_status"] or "available"))


def _item_from_mapping(row: dict[str, Any], *, local_status: str) -> RunHistoryItem:
    return RunHistoryItem(
        run_id=str(row["run_id"]),
        cache_key=str(row["cache_key"]),
        project_id=str(row["project_id"]),
        project_name=row.get("project_name"),
        project_root=row.get("project_root"),
        task_root=row.get("task_root"),
        active_file=row.get("active_file"),
        prompt_preview=str(row.get("prompt_preview") or "(empty prompt)"),
        policy=str(row.get("policy") or "unknown"),
        family=str(row.get("family") or "unknown"),
        created_at=str(row.get("created_at") or utc_now_iso()),
        updated_at=str(row.get("updated_at") or row.get("created_at") or utc_now_iso()),
        visible_passes=int(row.get("visible_passes") or 0),
        hidden_passes=int(row.get("hidden_passes") or 0),
        compile_successes=int(row.get("compile_successes") or 0),
        total_reward=float(row.get("total_reward") or 0.0),
        elapsed_s=row.get("elapsed_s"),
        llm_requests=int(row.get("llm_requests") or 0),
        local_status=local_status,  # type: ignore[arg-type]
        cloud_status=str(row.get("cloud_status") or "unknown"),  # type: ignore[arg-type]
        cloud_object_path=row.get("cloud_object_path"),
        package_sha256=row.get("package_sha256"),
    )


def _cloud_row(item: RunHistoryItem) -> dict[str, Any]:
    return item.model_dump(mode="json")


def _safe_error(exc: Exception) -> str:
    return str(exc).replace("\n", " ")[:500]
