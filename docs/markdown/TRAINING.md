# Training Flow

This repo supports offline bootstrap training first, then optional online fine-tuning.

## 1. Collect synthetic data

Use `scripts/data/collect_data.py` to generate search-control trajectories.

Before reporting results, generate a deterministic task-level split stratified by difficulty:

```bash
conda run -n hyperreasoning python scripts/data/make_splits.py \
  --tasks-root data/generated_tasks \
  --out-dir data/splits \
  --eval-count 10 \
  --seed 123 \
  --train-name train_90.txt \
  --eval-name eval_10.txt
```

Common run types:

- bulk heuristic
  - many episodes
  - low verification budget
  - cheap and broad
- hybrid enrichment
  - fewer episodes
  - moderate verification budget
  - better reward grounding

## 2. Train offline Rainbow

Use `scripts/train/train_rainbow.py`.

Example:

```bash
conda run -n hyperreasoning python scripts/train/train_rainbow.py \
  --run-dirs artifacts/synthetic/heuristic_bulk_v1 artifacts/synthetic/hybrid_enrichment_v1 artifacts/synthetic/hybrid_enrichment_v2 \
  --offline-updates 5000 \
  --batch-size 128 \
  --buffer-capacity 100000 \
  --experiment-name rainbow_offline_v1
```

Optional online fine-tuning can use the train split manifest:

```bash
conda run -n hyperreasoning python scripts/train/train_rainbow.py \
  --run-dirs artifacts/synthetic/heuristic_bulk_v1 artifacts/synthetic/hybrid_enrichment_v1 artifacts/synthetic/hybrid_enrichment_v2 \
  --online-episodes 50 \
  --task-manifest data/splits/train_90.txt \
  --experiment-name rainbow_online_v1
```

What the trainer does:

1. loads one or more synthetic run directories
2. normalizes transitions into the canonical offline RL schema
3. fits the `StateEncoder` vocabulary on observed states
4. encodes transitions into fixed-width numeric tensors
5. warms the prioritized replay buffer from offline data
6. trains the masked C51 Rainbow agent
7. saves:
   - model checkpoint
   - state encoder metadata
   - training config
   - TensorBoard logs

Outputs are saved under:

```text
artifacts/models/<experiment_name>_<timestamp>/
```

For one-off smoke/debug runs, prefer an explicit output under:

```text
artifacts/smoke/
```

## 3. Evaluate policies

Use `scripts/eval/eval_baselines.py`.

Policies:

- `random`
- `heuristic`
- `rainbow`
- `all`

Example:

```bash
conda run -n hyperreasoning python scripts/eval/eval_baselines.py \
  --task-manifest data/splits/eval_10.txt \
  --num-tasks 17 \
  --episodes-per-task 1 \
  --policy all \
  --checkpoint artifacts/models/<run_name>/checkpoint.pt
```

For final comparisons, use the held-out eval split and prefer the saved `best.pt` checkpoint over `latest.pt`. The current main run showed that `latest.pt` can be meaningfully worse than `best.pt`, so the best-checkpoint artifact is the one that should be used for reporting.

Recommended final eval pattern:

```bash
conda run -n hyperreasoning python scripts/eval/eval_baselines.py \
  --task-manifest data/splits/eval_10.txt \
  --num-tasks 10 \
  --episodes-per-task 1 \
  --policy all \
  --checkpoint artifacts/models/<run_name>/best.pt \
  --run-tests \
  --max-verified-plans-per-task 1
```

## 4. Optional online fine-tuning

`scripts/train/train_rainbow.py` also supports online episodes by setting:

- `--online-episodes > 0`

In that mode:

- offline replay is loaded first if provided
- the agent interacts with the canonical DSL environment
- online transitions are added into replay
- updates continue from mixed replay

## Metrics You Should Watch

During training:

- loss
- mean Q value
- mean sampled reward
- replay size

During evaluation:

- mean return
- mean episode length
- compile success rate
- visible pass rate

## Current Limitations

- `REPAIR_FROM_FEEDBACK` is currently a deterministic proxy action rather than a fully learned repair subsystem.
- Prioritized replay is a simple proportional implementation, not a sum-tree.
- The current environment wrapper uses the existing search-control collector/runtime rather than a fully separate rewritten engine.

These are deliberate v1 tradeoffs to keep the system runnable end to end.

## Current Observed Result Pattern

On the current held-out eval split:
- `best.pt` can be substantially better than `latest.pt`
- compile and visible-pass metrics can exceed `1.0` because the current logger counts successful events across episodes rather than clamping to one success per episode

So for future runs:
- track `best.pt`
- do not assume the latest checkpoint is the strongest checkpoint
- compare Rainbow against both random and heuristic on the held-out split
