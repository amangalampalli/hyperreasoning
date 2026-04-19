# JetBrains Plugin Strategy

The JetBrains plugin should focus on **inference-time use and visualization**,
not training.

## Intended Responsibilities

- accept a coding task inside the IDE
- trigger DSL search / branch-control runs in the backend
- visualize:
  - candidate DSL branches
  - selected actions
  - verifier outcomes
  - final patch
- optionally compare heuristic vs Rainbow on the same task

## Suggested Integration Strategy

Phase 1:
- task-oriented integration
- trigger a backend search run for the current task through the local FastAPI backend
- render the resulting task-local tree / verifier / patch outputs

Phase 2:
- optional local HTTP bridge for live backend communication and streaming updates

## Scaffold Location

The plugin scaffold lives under:

```text
jetbrains-plugin/
```

This keeps the plugin and Python backend in the same repo for hackathon speed
and easier synchronized iteration.
