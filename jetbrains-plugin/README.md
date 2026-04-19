# Hyperreasoning JetBrains Plugin

Native IntelliJ/PyCharm plugin for task-time Hyperreasoning search runs.

The plugin is the main hackathon demo surface. It lets a user run search from an open task, compare strategies, and inspect a live search graph inside the IDE.

## What It Does

- Reads the active task or current editor file.
- Submits task runs to the local FastAPI backend.
- Runs individual strategies:
  - `Rainbow`
  - `Heuristic`
  - `1-Shot LLM`
  - `Random`
- Runs `Compare` across the main strategies.
- Streams graph events while async backend jobs run.
- Shows ranked result cards and verifier summaries.
- Visualizes the search in a native Swing tool window.

## Search Graph

The `Search Graph` tab has two modes.

### Tree View

Default mode. Shows the full materialized search tree for the selected run.

Visual cues:

- root node
- active/revealed candidates
- currently expanding candidates
- pruned candidates
- best-path edge highlight
- compile/test failure nodes
- success node

Interactions:

- click a node to open details
- drag to pan
- mouse/trackpad wheel to zoom
- `Fit View` to frame the graph
- `Expand` to open a larger graph window
- strategy buttons to switch the selected recorded strategy after a run

### Guided Path

Toggle `Guided path` to replace the tree with an animated decision timeline.

The timeline shows the step-by-step decisions the controller made:

- root
- selected candidates
- expansion/reveal decisions
- backtracking/revisit steps
- transition labels between decisions
- repeated visits as separate timeline cards

Timeline cards are draggable and spring back into place for a more interactive demo view.

## Local Backend

The plugin expects the backend at:

```text
http://127.0.0.1:8765
```

Start it from the repo root:

```bash
conda run -n hyperreasoning python scripts/serve/run_backend.py \
  --host 127.0.0.1 \
  --port 8765 \
  --llm-base-url http://127.0.0.1:8080
```

The backend API and graph event contract are documented in:

- [Backend service](../docs/markdown/BACKEND.md)
- [Search graph event contract](../docs/SEARCH_GRAPH_EVENTS.md)

## Build

From this folder:

```bash
JAVA_HOME='/Applications/IntelliJ IDEA.app/Contents/jbr/Contents/Home' ./gradlew build
```

## Run in Sandbox IDE

```bash
JAVA_HOME='/Applications/IntelliJ IDEA.app/Contents/jbr/Contents/Home' ./gradlew runIde
```

## Demo Checklist

1. Start the backend.
2. Open a generated task or demo task in the sandbox IDE.
3. Open the `Hyperreasoning` tool window.
4. Confirm backend health.
5. Run `Rainbow`.
6. Show live graph updates.
7. Toggle `Guided path`.
8. Expand the graph window.
9. Run `Compare`.
10. Switch between strategy buttons and show consistent tree rendering.

For the full demo script, see [docs/DEMO_RUNBOOK.md](../docs/DEMO_RUNBOOK.md).
