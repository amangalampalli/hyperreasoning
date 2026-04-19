# Search Graph Event Contract

The JetBrains plugin renders the search graph from incremental events emitted by the backend during async task runs.

The graph event stream is intentionally UI-oriented. It describes what the IDE should display, not every low-level runtime object.

## Transport

Async status endpoints return graph events with a cursor:

```http
GET /api/task/run_async/{job_id}?graph_event_cursor=<cursor>
GET /api/task/compare_async/{job_id}?graph_event_cursor=<cursor>
```

Responses include:

```json
{
  "graph_events": [],
  "graph_event_cursor": 12
}
```

The plugin sends the previous `graph_event_cursor` on the next poll. The backend returns only events after that cursor.

## Event Envelope

All graph events include:

```json
{
  "type": "node_created",
  "run_id": "job-id:policy"
}
```

`run_id` identifies the graph stream. A compare run may emit separate policy-specific run ids.

## Event Types

### `search_started`

Starts or resets a graph stream.

```json
{
  "type": "search_started",
  "run_id": "abc:rainbow",
  "title": "Rainbow Search",
  "subtitle": "Live backend search events are streaming into the graph."
}
```

### `search_reset`

Clears the graph state for a run.

```json
{
  "type": "search_reset",
  "run_id": "abc:rainbow"
}
```

### `node_created`

Creates a candidate/search node.

```json
{
  "type": "node_created",
  "run_id": "abc:rainbow",
  "node": {
    "id": "bank_00004",
    "parentId": "bank_00001",
    "depth": 2,
    "title": "Minimal Patch",
    "shortSummary": "minimal_patch | cache_invalidation",
    "dslSummary": "strategy=minimal_patch\nfiles=cache.py\nchecks=visible_tests",
    "patchSummary": "Targets cache.py",
    "rationaleSummary": "Focus invalidation edge case",
    "childIndex": 1,
    "childCount": 3,
    "status": "IDLE",
    "createdOrder": 4,
    "createdAtLabel": null,
    "rawMetadataJson": null
  }
}
```

Recognized node statuses:

- `ROOT`
- `IDLE`
- `ACTIVE`
- `EXPANDING`
- `PRUNED`
- `SUCCESS`
- `FAILED_COMPILE`
- `FAILED_TEST`
- `FAILED_RUNTIME`

### `edge_created`

Creates a parent-to-child edge.

```json
{
  "type": "edge_created",
  "run_id": "abc:rainbow",
  "parent_id": "bank_00001",
  "child_id": "bank_00004",
  "action_label": "SELECT_CHILD_0"
}
```

### `node_scored`

Adds ranking/model scores to a node.

```json
{
  "type": "node_scored",
  "run_id": "abc:rainbow",
  "score": {
    "id": "bank_00004",
    "score": 0.72,
    "rank": 1,
    "qValue": 0.51,
    "heuristicScore": 0.72
  }
}
```

### `node_status_changed`

Changes status and optional compile/test/runtime metadata.

```json
{
  "type": "node_status_changed",
  "run_id": "abc:rainbow",
  "node_id": "bank_00004",
  "status": "EXPANDING",
  "terminal_summary": "selected child 1",
  "compile_status": null,
  "test_status": null,
  "runtime_status": null
}
```

### `node_pruned`

Marks a node as pruned.

```json
{
  "type": "node_pruned",
  "run_id": "abc:rainbow",
  "node_id": "bank_00008",
  "reason": "Visible during search but never promoted"
}
```

### `best_path_updated`

Updates the current best path.

```json
{
  "type": "best_path_updated",
  "run_id": "abc:rainbow",
  "node_ids": ["root", "bank_00001", "bank_00004"]
}
```

The tree view highlights the corresponding tree edges. The guided path uses decision events and node status events to build the timeline.

### `search_finished`

Marks the stream as complete.

```json
{
  "type": "search_finished",
  "run_id": "abc:rainbow",
  "terminal_node_id": "bank_00004",
  "success": true,
  "summary": "Visible tests passed for the best node."
}
```

## Plugin Reducer

The plugin normalizes events into:

- nodes keyed by id
- edges keyed by `parent->child`
- best-path ids
- selected node id
- decision trail
- run status counters

The implementation lives in:

```text
jetbrains-plugin/src/main/kotlin/com/hyperreasoning/intellij/toolwindow/SearchGraphEvents.kt
```

The backend-to-plugin JSON adapter lives in:

```text
jetbrains-plugin/src/main/kotlin/com/hyperreasoning/intellij/toolwindow/SearchGraphBackendAdapter.kt
```

## Backend Emitters

The current backend emits graph events from:

```text
backend/jobs.py
```

The search runtime provides graph metadata from:

```text
env/dsl_env.py
```

When adding new graph metadata, prefer adding fields to the event payload rather than teaching the UI to parse unrelated backend internals.
