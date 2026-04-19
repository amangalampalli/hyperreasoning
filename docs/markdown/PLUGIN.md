# JetBrains Plugin

The JetBrains plugin is the main task-time inference and visualization surface.

It is responsible for:

- collecting task context from the open IDE project
- submitting backend task runs
- streaming graph events during async jobs
- rendering the search tree and guided decision path
- showing verifier and strategy summaries
- comparing Rainbow, heuristic, random, and one-shot strategies

## Location

```text
jetbrains-plugin/
```

## Current UI

The tool window contains:

- task prompt/context area
- visible/hidden test toggles
- strategy run buttons
- backend health check
- live progress card
- result summary cards
- `Search Graph` tab

The search graph supports:

- full tree view
- guided decision timeline
- node selection and details
- pan/zoom
- fit-to-view
- expanded graph window
- strategy switching after compare runs

## Backend Integration

The plugin talks to the local FastAPI backend:

```text
http://127.0.0.1:8765
```

Async runs stream graph events through job-status polling. See:

- [Backend service](BACKEND.md)
- [Search graph event contract](../SEARCH_GRAPH_EVENTS.md)

## Demo

Use the root demo runbook:

```text
docs/DEMO_RUNBOOK.md
```

## Build

```bash
cd jetbrains-plugin
JAVA_HOME='/Applications/IntelliJ IDEA.app/Contents/jbr/Contents/Home' ./gradlew build
```

## Run Sandbox IDE

```bash
cd jetbrains-plugin
JAVA_HOME='/Applications/IntelliJ IDEA.app/Contents/jbr/Contents/Home' ./gradlew runIde
```
