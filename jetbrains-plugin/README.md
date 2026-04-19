# JetBrains Plugin Scaffold

This folder contains a minimal IntelliJ/PyCharm plugin scaffold for the
hackathon demo.

The plugin should focus on **task-time inference and visualization**, not
training.

The intended role is:

- accept a coding task from the user
- trigger backend DSL search / branch control for that task
- visualize candidate branches and selected actions
- show verifier outcomes on that task
- present the final patch/diff inside the IDE
- optionally compare heuristic vs Rainbow on the same task

## Current State

This is a scaffold, not a completed plugin.

Included:
- Gradle/Kotlin IntelliJ plugin project structure
- plugin metadata (`plugin.xml`)
- a tool window factory
- placeholder panel and service classes
- explicit task actions in the tool window:
  - `Use Current File`
  - `Health`
  - `Run Search (Heuristic)`
  - `Run Search (Rainbow)`
  - `Compare Strategies`

Not included yet:
- backend process management
- file watching or HTTP polling
- tree visualization
- patch application
- heuristic vs Rainbow task comparison UI

## Suggested Next Implementation Steps

1. Add a backend connection strategy.
   - current approach: call the local FastAPI backend for single-task search runs
2. Implement the main tool window UI.
  - Task input area
  - Search tree / branch list
  - Verifier summary / final patch view
3. Add actions:
   - Run Search
   - Compare Heuristic vs Rainbow
   - Apply Patch
4. Add a small settings panel for locating the backend repo or service endpoint

## Development Notes

This scaffold targets the modern Gradle plugin layout described in the JetBrains
plugin SDK docs for Gradle-based projects.

Run from this folder with Gradle:

```bash
./gradlew runIde
```

or from IntelliJ IDEA as a Gradle project.

The plugin expects the local backend to be available at:

```text
http://127.0.0.1:8765
```
