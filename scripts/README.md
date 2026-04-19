# Scripts Layout

The `scripts/` directory is grouped by purpose:

- `scripts/data/`
  - data collection and task-generation utilities
- `scripts/train/`
  - training entrypoints
- `scripts/eval/`
  - evaluation and run-summary entrypoints
- `scripts/debug/`
  - legacy debugging helpers
- `scripts/serve/`
  - local model/server helpers

Primary entrypoints:
- `scripts/data/collect_data.py`
- `scripts/train/train_rainbow.py`
- `scripts/eval/eval_baselines.py`
- `scripts/run_eval.py`
- `scripts/eval/run_codex_benchmark.py`
