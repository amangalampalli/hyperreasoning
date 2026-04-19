# Local Backend Service

The JetBrains plugin should talk to a small local Python backend rather than
calling `llama.cpp` directly.

## Why

- keeps model/prompt logic out of the plugin
- makes backend swaps easier later
- centralizes DSL search, verifier, and Rainbow inference
- keeps the plugin thin and UI-focused

## Current Backend

Location:

```text
backend/
```

Entry point:

```text
scripts/serve/run_backend.py
```

## Endpoints

- `GET /health`
  - backend and LLM reachability summary
- `POST /api/task/run`
  - run a single task with one strategy (`random`, `heuristic`, `rainbow`, or `oneshot`)
- `POST /api/task/compare`
  - compare multiple strategies on the same task
- `POST /api/task/run_async`
  - start an async single-strategy run
- `GET /api/task/run_async/{job_id}`
  - poll task status, progress, result, and graph events
- `POST /api/task/compare_async`
  - start an async compare run
- `GET /api/task/compare_async/{job_id}`
  - poll compare status, progress, result, and graph events
- `GET /api/runs`
  - list cached project runs from `.hyper/` and, when configured, Supabase metadata
- `GET /api/runs/{run_id}`
  - load a cached run package locally or restore it from Supabase Storage
- `POST /api/runs/sync`
  - retry pending cloud uploads and restore missing local packages from Supabase

## Run Cache and Supabase Backup

The backend stores IDE run packages under the opened project root:

```text
<project>/.hyper/
```

Cloud backup is optional. Add a local `.env` file at the repository root:

```bash
HYPERREASONING_SUPABASE_URL=https://your-project-ref.supabase.co
HYPERREASONING_SUPABASE_KEY=your-backend-only-secret-or-service-role-key
HYPERREASONING_SUPABASE_BUCKET=hyperreasoning-runs
HYPERREASONING_SUPABASE_RUNS_TABLE=hyperreasoning_runs
```

Create the Supabase Storage bucket named by `HYPERREASONING_SUPABASE_BUCKET`
and a metadata table with columns matching the run history payload:

```sql
create table if not exists hyperreasoning_runs (
  run_id text primary key,
  cache_key text not null,
  project_id text not null,
  project_name text,
  project_root text,
  task_root text,
  active_file text,
  prompt_preview text not null,
  policy text not null,
  family text not null,
  created_at timestamptz not null,
  updated_at timestamptz not null,
  visible_passes integer not null default 0,
  hidden_passes integer not null default 0,
  compile_successes integer not null default 0,
  total_reward double precision not null default 0,
  elapsed_s double precision,
  llm_requests integer not null default 0,
  local_status text not null default 'available',
  cloud_status text not null default 'unknown',
  cloud_object_path text,
  package_sha256 text
);

create index if not exists hyperreasoning_runs_project_created_idx
  on hyperreasoning_runs (project_id, created_at desc);
```

Use a backend-only Supabase key. Do not put Supabase secrets in the JetBrains
plugin settings.

## Suggested Demo Flow

1. Open the plugin tool window in PyCharm
2. Enter a prompt
3. Use the current editor file as the target file
4. Trigger:
   - `Run Heuristic`
   - or `Run Rainbow`
5. Show:
   - live candidate tree
   - guided decision path
   - verifier outcomes
   - strategy ranking and final patch summary

For the full walkthrough, see [Demo runbook](../DEMO_RUNBOOK.md).

## Search Graph Events

Async status responses include graph events and a cursor:

```json
{
  "graph_events": [],
  "graph_event_cursor": 12
}
```

The plugin passes `graph_event_cursor` back on the next poll to receive only new graph events.

See [Search graph event contract](../SEARCH_GRAPH_EVENTS.md).

## Run Command

```bash
conda run -n hyperreasoning python scripts/serve/run_backend.py \
  --host 127.0.0.1 \
  --port 8765 \
  --llm-base-url http://127.0.0.1:8080
```
