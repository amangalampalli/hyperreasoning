# Task Splits

This folder contains reproducible task-level train/eval manifests.

Current split:

- `train_90.txt`
- `eval_10.txt`

Current defaults:

- `160` train tasks
- `10` eval tasks
- eval split stratified by difficulty

These manifests list task directory paths relative to the repo root.

Use them with:

```bash
conda run -n hyperreasoning python scripts/eval/eval_baselines.py \
  --task-manifest data/splits/eval_10.txt \
  --policy all \
  --checkpoint artifacts/models/<run_name>/checkpoint.pt
```

For online fine-tuning on the training split:

```bash
conda run -n hyperreasoning python scripts/train/train_rainbow.py \
  --run-dirs artifacts/synthetic/heuristic_bulk_v1 artifacts/synthetic/hybrid_enrichment_v1 artifacts/synthetic/hybrid_enrichment_v2 \
  --online-episodes 50 \
  --task-manifest data/splits/train_90.txt
```
