# Runbook

## Bulk synthetic collection

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

## Generate train/eval split

```bash
conda run -n hyperreasoning python scripts/data/make_splits.py \
  --tasks-root data/generated_tasks \
  --out-dir data/splits \
  --eval-count 10 \
  --seed 123 \
  --train-name train_90.txt \
  --eval-name eval_10.txt
```

## Resume a collection run

```bash
conda run -n hyperreasoning python scripts/data/collect_data.py \
  --tasks-root data/generated_tasks \
  --num-tasks 150 \
  --episodes-per-task 40 \
  --max-steps 6 \
  --proposal-source heuristic \
  --max-verified-plans-per-task 1 \
  --max-llm-workers 2 \
  --run-id heuristic_bulk_v1 \
  --resume
```

## Enrichment collection

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

## Offline Rainbow training

```bash
conda run -n hyperreasoning python scripts/train/train_rainbow.py \
  --run-dirs artifacts/synthetic/heuristic_bulk_v1 artifacts/synthetic/hybrid_enrichment_v1 \
  --offline-updates 5000 \
  --batch-size 128 \
  --buffer-capacity 100000 \
  --experiment-name rainbow_offline_v1
```

## Baseline / policy evaluation

```bash
conda run -n hyperreasoning python scripts/eval/eval_baselines.py \
  --tasks-root data/generated_tasks/medium \
  --num-tasks 10 \
  --episodes-per-task 1 \
  --policy all \
  --checkpoint artifacts/models/<run_name>/checkpoint.pt
```

For reporting or demos, prefer the `best.pt` checkpoint:

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

Notes:
- `best.pt` is often meaningfully stronger than `latest.pt`
- compile/pass rates in current logs count successful events across eval episodes, so values can exceed `1.0`

## Smoke output convention

Use `artifacts/smoke/` for temporary validation outputs, for example:
- `artifacts/smoke/model_smoke/`
- `artifacts/smoke/model_smoke_refactor/`

## Focused tests

```bash
conda run -n hyperreasoning python -m pytest -q tests
```
