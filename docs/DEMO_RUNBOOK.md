# Demo Runbook

This runbook is for the JetBrains hackathon demo flow.

## Prerequisites

- Conda environment: `hyperreasoning`
- Local LLM endpoint reachable by the backend
- IntelliJ IDEA or PyCharm available locally
- Repository opened as the current workspace

## Start Backend

From the repo root:

```bash
conda run -n hyperreasoning python scripts/serve/run_backend.py \
  --host 127.0.0.1 \
  --port 8765 \
  --llm-base-url http://127.0.0.1:8080
```

The plugin expects:

```text
http://127.0.0.1:8765
```

## Start Plugin Sandbox

From `jetbrains-plugin/`:

```bash
JAVA_HOME='/Applications/IntelliJ IDEA.app/Contents/jbr/Contents/Home' ./gradlew runIde
```

## Demo Flow

1. Open a demo task or generated task in the sandbox IDE.
2. Open the `Hyperreasoning` tool window.
3. Click `Check Backend`.
4. Confirm the backend health status is OK.
5. Run `Rainbow`.
6. Open the `Search Graph` tab.
7. Show the live tree:
   - nodes appear as candidates are revealed
   - active/expanding/pruned/success states use distinct styling
   - best-path edges are highlighted
8. Click nodes to show details:
   - node id
   - parent id
   - depth
   - status
   - score/heuristic
   - DSL summary
   - patch summary
   - compile/test status
9. Toggle `Guided path`.
10. Show the animated decision timeline:
    - root
    - selected candidates
    - reveal/refine/backtrack decisions
    - repeated visits as separate cards
    - transition labels inside cards
11. Use `Expand` for a larger graph window.
12. Run `Compare`.
13. Switch between strategy buttons:
    - `Rainbow`
    - `Heuristic`
    - `1-Shot LLM`
14. Show ranked result cards in `Summary`.

## What To Say

Suggested narration:

- "We search over structured implementation plans instead of raw edits."
- "The controller ranks branches and decides which candidates to expand or verify."
- "The tree view shows the candidate search space for this task."
- "The guided path shows the decisions the controller made step by step."
- "Verifier outcomes feed back into the graph: compile failures, test failures, and successful candidates are visible immediately."
- "Compare runs the same task through multiple strategies and ranks the outcomes."

## Important Controls

- `Fit View`: frame the full tree.
- `Expand`: open a larger graph window.
- `Guided path`: switch between full tree and decision timeline.
- `Auto-follow`: follow new graph activity during live runs.
- Strategy buttons: switch the selected recorded strategy after a run.

## Troubleshooting

Backend unavailable:

- Check the backend terminal.
- Run `Check Backend` in the plugin.
- Confirm port `8765`.

Graph does not update:

- Confirm the run uses async backend endpoints.
- Confirm `graph_events` are returned in async status payloads.
- Check [Search graph event contract](SEARCH_GRAPH_EVENTS.md).

Plugin changes not visible:

- Rebuild the plugin.
- Restart the sandbox IDE.
- Confirm you are testing the sandbox from the current checkout.

```bash
cd jetbrains-plugin
JAVA_HOME='/Applications/IntelliJ IDEA.app/Contents/jbr/Contents/Home' ./gradlew build
JAVA_HOME='/Applications/IntelliJ IDEA.app/Contents/jbr/Contents/Home' ./gradlew runIde
```
