# Hyperreasoning

Hyperreasoning is a JetBrains/Codex hackathon project for DSL-guided code search with a Rainbow-style branch controller.

The core idea is to search over structured implementation plans instead of raw code edits:

- generate compact candidate plans in a DSL
- rank and traverse candidate branches with heuristic or Rainbow policies
- compile and verify promising plans against task tests
- surface the search process in a native JetBrains tool window
- compare Rainbow against heuristic, random, and one-shot LLM baselines

## Current Status

The repo contains an end-to-end demo stack:

- Python search-control environment and verifier
- plan DSL proposal and compiler interfaces
- local FastAPI backend for IDE-triggered task runs
- native JetBrains plugin with live search-graph visualization
- offline dataset collection and Rainbow training scripts
- held-out evaluation reports and benchmark artifacts

The current implementation is optimized for clarity and hackathon demo reliability.

## Results

Primary held-out eval split: `data/splits/eval_10.txt`

Current headline result:

| Method | Solve rate | Mean tests passed | Mean time | Mean tokens |
| --- | ---: | ---: | ---: | ---: |
| Rainbow | 80.0% | 80.0% | 15,353 ms | 1,694 |
| Heuristic | 40.0% | 40.0% | 23,144 ms | 2,340 |
| Random | 30.0% | 37.5% | 23,009 ms | 2,342 |
| One-shot | 30.0% | 37.5% | 31,016 ms | 3,649 |

See:

- [Eval results](docs/markdown/results.md)
- [Codex benchmark comparison](docs/codex_benchmark/results.md)
- [Summary metrics CSV](docs/summary_metrics.csv)
- [Solve-rate chart](docs/png/solve_rate.png)
- [Rainbow vs heuristic chart](docs/png/rainbow_vs_heuristic_delta.png)
- [Rainbow vs one-shot chart](docs/png/rainbow_vs_one_shot_delta.png)

Best observed Rainbow checkpoint:

```text
artifacts/models/rainbow_offline_v1_1776237078/best.pt
```

## Demo

The primary demo surface is the JetBrains plugin in `jetbrains-plugin/`.

Demo flow:

1. Start the local backend.
2. Open the plugin tool window in IntelliJ/PyCharm.
3. Open a task file or task folder.
4. Run `Rainbow`, `Heuristic`, `1-Shot LLM`, or `Compare`.
5. Show the live search graph, verifier status, and ranked results.

Useful docs:

- [Demo runbook](docs/DEMO_RUNBOOK.md)
- [JetBrains plugin guide](jetbrains-plugin/README.md)
- [Search graph event contract](docs/SEARCH_GRAPH_EVENTS.md)
- [Backend service](docs/markdown/BACKEND.md)

## Repo Map

- `backend/` - FastAPI bridge for task-time search runs and live graph events
- `jetbrains-plugin/` - native JetBrains plugin and search visualization UI
- `env/` - search-control runtime, verifier integration, rewards, state encoder
- `llm/` - DSL proposal, compiler, repair, and prompt utilities
- `rl/` - Rainbow/C51 implementation
- `models/` - neural network components
- `data/` - task store, splits, transition schemas, replay dataset helpers
- `scripts/` - data collection, training, eval, serving, and debug entrypoints
- `tests/` - backend, env, data, llm, and rl tests
- `docs/` - reports, event contract, backend/plugin notes, charts, CSVs
- `artifacts/` - generated synthetic data, model checkpoints, and run outputs
- `agents/` - operator runbook and planning notes

## Main Commands

### Start the backend

```bash
conda run -n hyperreasoning python scripts/serve/run_backend.py \
  --host 127.0.0.1 \
  --port 8765 \
  --llm-base-url http://127.0.0.1:8080
```

### Run tests

```bash
conda run -n hyperreasoning python -m pytest -q tests
```

### Build the JetBrains plugin

```bash
cd jetbrains-plugin
JAVA_HOME='/Applications/IntelliJ IDEA.app/Contents/jbr/Contents/Home' ./gradlew build
```

### Run the plugin sandbox

```bash
cd jetbrains-plugin
JAVA_HOME='/Applications/IntelliJ IDEA.app/Contents/jbr/Contents/Home' ./gradlew runIde
```

### Train Rainbow offline

```bash
conda run -n hyperreasoning python scripts/train/train_rainbow.py \
  --run-dirs artifacts/synthetic/heuristic_bulk_v1 artifacts/synthetic/hybrid_enrichment_v1 artifacts/synthetic/hybrid_enrichment_v2 \
  --offline-updates 5000 \
  --batch-size 128 \
  --buffer-capacity 100000 \
  --experiment-name rainbow_offline_v1
```

### Evaluate baselines

```bash
conda run -n hyperreasoning python scripts/eval/eval_baselines.py \
  --task-manifest data/splits/eval_10.txt \
  --num-tasks 10 \
  --episodes-per-task 1 \
  --policy all \
  --checkpoint artifacts/models/rainbow_offline_v1_1776237078/best.pt \
  --run-tests \
  --max-verified-plans-per-task 1
```

## Data and Training Notes

- Synthetic data lives under `artifacts/synthetic/<run_id>/`.
- Checkpoints live under `artifacts/models/`.
- Split manifests live under `data/splits/`.
- Eval charts and CSVs live under `docs/`.
- The trainer consumes `dataset.jsonl` files from collected runs.
- Action masking is part of the transition contract.

## Documentation Index

- [Demo runbook](docs/DEMO_RUNBOOK.md)
- [Search graph event contract](docs/SEARCH_GRAPH_EVENTS.md)
- [JetBrains plugin](jetbrains-plugin/README.md)
- [Backend service](docs/markdown/BACKEND.md)
- [Dataset format](docs/markdown/DATASET.md)
- [Training flow](docs/markdown/TRAINING.md)
- [Plugin strategy notes](docs/markdown/PLUGIN.md)
- [Eval results](docs/markdown/results.md)
- [Codex benchmark results](docs/codex_benchmark/results.md)
- [Operator runbook](agents/RUNBOOK.md)
