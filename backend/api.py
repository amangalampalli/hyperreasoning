"""FastAPI app for local plugin/backend communication."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request

from backend.config import load_backend_config
from backend.jobs import AsyncJobStore
from backend.models import (
    AsyncJobAcceptedResponse,
    ClientContext,
    CompareJobStatusResponse,
    CompareStrategiesRequest,
    CompareStrategiesResponse,
    HealthResponse,
    RunHistoryResponse,
    RunLoadResponse,
    RunSyncRequest,
    RunSyncResponse,
    TaskRunJobStatusResponse,
    TaskRunRequest,
    TaskRunResponse,
)
from backend.run_cache import list_run_history, load_cached_run, sync_runs
from backend.service import compare_strategies, llm_health, run_task_request


def create_app(llm_base_url: str = "http://127.0.0.1:8080") -> FastAPI:
    app = FastAPI(title="Hyperreasoning Backend", version="0.1.0")
    backend_config = load_backend_config()
    jobs = AsyncJobStore(backend_config=backend_config)

    async def _read_json_body(request: Request) -> dict:
        raw = await request.body()
        if not raw:
            raise HTTPException(status_code=422, detail="Request body was empty")
        try:
            payload = await request.json()
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Could not parse JSON body: {exc}") from exc
        if not isinstance(payload, dict):
            raise HTTPException(status_code=422, detail="Expected JSON object body")
        return payload

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return llm_health(llm_base_url)

    @app.post("/api/task/run", response_model=TaskRunResponse)
    async def run_task(request: Request) -> TaskRunResponse:
        try:
            payload = await _read_json_body(request)
            parsed = TaskRunRequest.model_validate(payload)
            return run_task_request(parsed, llm_base_url=llm_base_url, backend_config=backend_config)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/api/task/compare", response_model=CompareStrategiesResponse)
    async def compare_task(request: Request) -> CompareStrategiesResponse:
        try:
            payload = await _read_json_body(request)
            parsed = CompareStrategiesRequest.model_validate(payload)
            return compare_strategies(parsed, llm_base_url=llm_base_url, backend_config=backend_config)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/api/task/run_async", response_model=AsyncJobAcceptedResponse)
    async def run_task_async(request: Request) -> AsyncJobAcceptedResponse:
        try:
            payload = await _read_json_body(request)
            parsed = TaskRunRequest.model_validate(payload)
            return jobs.start_task_run(parsed, llm_base_url=llm_base_url)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.get("/api/task/run_async/{job_id}", response_model=TaskRunJobStatusResponse)
    async def get_task_run_async(job_id: str) -> TaskRunJobStatusResponse:
        try:
            return jobs.get_task_run(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown task run job: {job_id}") from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/api/task/compare_async", response_model=AsyncJobAcceptedResponse)
    async def compare_task_async(request: Request) -> AsyncJobAcceptedResponse:
        try:
            payload = await _read_json_body(request)
            parsed = CompareStrategiesRequest.model_validate(payload)
            return jobs.start_compare(parsed, llm_base_url=llm_base_url)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.get("/api/task/compare_async/{job_id}", response_model=CompareJobStatusResponse)
    async def get_compare_task_async(job_id: str) -> CompareJobStatusResponse:
        try:
            return jobs.get_compare(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown compare job: {job_id}") from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.get("/api/runs", response_model=RunHistoryResponse)
    async def list_runs(
        project_id: str,
        project_root: str | None = None,
        project_name: str | None = None,
        task_root: str | None = None,
        active_file: str | None = None,
        limit: int = 50,
        offset: int = 0,
        query: str | None = None,
    ) -> RunHistoryResponse:
        try:
            context = ClientContext(
                project_id=project_id,
                project_name=project_name,
                project_root=project_root,
                task_root=task_root,
                active_file=active_file,
            )
            return list_run_history(
                context,
                limit=limit,
                offset=offset,
                query=query,
                backend_config=backend_config,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.get("/api/runs/{run_id}", response_model=RunLoadResponse)
    async def get_run(
        run_id: str,
        project_id: str | None = None,
        project_root: str | None = None,
        project_name: str | None = None,
        task_root: str | None = None,
        active_file: str | None = None,
    ) -> RunLoadResponse:
        try:
            context = (
                ClientContext(
                    project_id=project_id,
                    project_name=project_name,
                    project_root=project_root,
                    task_root=task_root,
                    active_file=active_file,
                )
                if project_id or project_root
                else None
            )
            return load_cached_run(run_id, context=context, backend_config=backend_config)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown cached run: {run_id}") from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/api/runs/sync", response_model=RunSyncResponse)
    async def sync_run_history(request: Request) -> RunSyncResponse:
        try:
            payload = await _read_json_body(request)
            parsed = RunSyncRequest.model_validate(payload)
            return sync_runs(parsed, backend_config=backend_config)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app
