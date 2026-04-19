# Hyperreasoning Repo Guidance

- Python environment: use `conda run -n hyperreasoning ...`.
- Synthetic offline RL data is written under `artifacts/synthetic/<run_id>/`.
- Main bulk data collection:
  - `conda run -n hyperreasoning python scripts/data/collect_data.py --tasks-root data/generated_tasks --num-tasks 150 --episodes-per-task 40 --max-steps 6 --proposal-source heuristic --max-verified-plans-per-task 1 --max-llm-workers 2 --run-id heuristic_bulk_v1`
- Resume a collection run:
  - add `--run-id <name> --resume`
- Unit tests:
  - `conda run -n hyperreasoning python -m pytest -q`
- Keep `env/` as the canonical runtime API.
- Prefer JSONL transition compatibility over schema redesign; current training input should continue to accept the existing collector output.
