# Hyperreasoning

Hyperreasoning is a JetBrains/Codex hackathon project for DSL-guided code search with a Rainbow-style RL controller.

The core idea is:

- generate compact implementation plans in a structured DSL
- search over those plans instead of searching directly in raw code
- use a discrete-action Rainbow controller to decide which branches to expand, refine, compile, backtrack from, or terminate
- ground reward in compile/test/verifier outcomes rather than subjective scoring

## Current Status

The repo now has a working end-to-end scaffold for:

- synthetic search-control data collection
- offline dataset loading from `artifacts/synthetic/*`
- a compact numeric state encoder
- a masked dueling C51 Rainbow agent
- offline training from collected transitions
- baseline evaluation against random and heuristic policies

The collection and training path is intentionally simple and debuggable rather than heavily optimized.

## Current Performance

Held-out evaluation is currently run on the deterministic `data/splits/eval_10.txt` split.

Best observed checkpoint from the main offline run:

```text
artifacts/models/rainbow_offline_v1_1776237078/best.pt
```

Observed held-out comparison on that split:

| Policy | Visible Passes | Compile Successes | Mean Return |
| --- | ---: | ---: | ---: |
| Random | 1 / 10 | 1 / 10 | 0.3655 |
| Heuristic | 5 / 10 | 5 / 10 | 1.8025 |
| Rainbow (`best.pt`) | 12 successful pass events across 10 episodes | 12 successful compile events across 10 episodes | 3.4755 |
| Rainbow (`latest.pt`) | 8 successful pass events across 10 episodes | 8 successful compile events across 10 episodes | 2.2985 |

Notes:

- The current eval metrics count successful compile/pass events across episodes, so Rainbow values can exceed `1.0` when multiple successful events happen within the same eval episode.
- `best.pt` significantly outperforms both the heuristic baseline and the final `latest.pt`, so it is the checkpoint that should be used for reporting and demos.

## Repo Map

- `env/`
  - canonical environment/runtime API
  - DSL env wrapper, reward helpers, verifier exports, state encoder
- `data/`
  - transition schema, task discovery, replay dataset loading, n-step reconstruction
- `rl/`
  - project-local Rainbow/C51 implementation, replay buffer, priorities, categorical projection
- `models/`
  - dueling categorical Q-network and small model helpers
- `llm/`
  - proposal/compiler/repair interfaces
- `scripts/`
  - organized by purpose under `data/`, `train/`, `eval/`, `debug/`, and `serve/`
- `backend/`
  - local FastAPI bridge for plugin/task-time search runs
- `tests/`
  - grouped tests under `env/`, `data/`, and `rl/`
  - backend API smoke tests under `tests/backend/`
- `artifacts/synthetic/`
  - synthetic trajectory runs used for offline bootstrap training
- `agents/`
  - implementation plan and operator runbook

## Main Commands

### Generate a deterministic stratified train/eval split

```bash
conda run -n hyperreasoning python scripts/data/make_splits.py \
  --tasks-root data/generated_tasks \
  --out-dir data/splits \
  --eval-count 10 \
  --seed 123 \
  --train-name train_90.txt \
  --eval-name eval_10.txt
```

### Bulk synthetic collection

```bash
conda run -n hyperreasoning python scripts/data/collect_data.py \
  --tasks-root data/generated_tasks \
  --num-tasks 150 \
  --episodes-per-task 40 \
  --max-steps 6 \
  --proposal-source heuristic \
  --max-verified-plans-per-task 1 \
  --max-llm-workers 2 \
  --run-id heuristic_bulk_v1
```

### Hybrid enrichment collection

```bash
conda run -n hyperreasoning python scripts/data/collect_data.py \
  --tasks-root data/generated_tasks \
  --num-tasks 150 \
  --episodes-per-task 10 \
  --max-steps 6 \
  --proposal-source hybrid \
  --max-verified-plans-per-task 2 \
  --max-llm-workers 2 \
  --run-tests \
  --allow-full-file-fallback \
  --run-id hybrid_enrichment_v1
```

### Offline Rainbow training

```bash
conda run -n hyperreasoning python scripts/train/train_rainbow.py \
  --run-dirs artifacts/synthetic/heuristic_bulk_v1 artifacts/synthetic/hybrid_enrichment_v1 artifacts/synthetic/hybrid_enrichment_v2 \
  --offline-updates 5000 \
  --batch-size 128 \
  --buffer-capacity 100000 \
  --experiment-name rainbow_offline_v1
```

### Baseline evaluation

```bash
conda run -n hyperreasoning python scripts/eval/eval_baselines.py \
  --tasks-root data/generated_tasks/medium \
  --num-tasks 10 \
  --episodes-per-task 1 \
  --policy all \
  --checkpoint artifacts/models/<run_name>/checkpoint.pt
```

### Local backend for the plugin

```bash
conda run -n hyperreasoning python scripts/serve/run_backend.py \
  --host 127.0.0.1 \
  --port 8765 \
  --llm-base-url http://127.0.0.1:8080
```

To compare the final held-out policies on the current split, prefer:

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

### Tests

```bash
conda run -n hyperreasoning python -m pytest -q tests
```

## Data + Training Notes

- Synthetic training data lives under `artifacts/synthetic/<run_id>/`.
- Each run writes:
  - `dataset.jsonl`
  - `run_summary.json`
  - per-task `plan_bank.json`
  - per-task `summary.json`
  - per-task `transitions.jsonl`
  - per-episode JSON trees
- The trainer consumes the `dataset.jsonl` files directly.
- Action masking is part of the stored transition contract and is used by both behavior and target action selection.
- Use `artifacts/smoke/` for one-off debug/eval smoke outputs only.
- Real training checkpoints default to `artifacts/models/`.

## More Docs

- [Dataset format](docs/markdown/DATASET.md)
- [Training flow](docs/markdown/TRAINING.md)
- [Backend service](docs/markdown/BACKEND.md)
- [Operator runbook](agents/RUNBOOK.md)
- [Execution plan](agents/PLANS.md)
