# Rainbow DSL Search Execution Plan

## Goal
Ship a working end-to-end Rainbow-style offline-to-online RL scaffold for the DSL search agent using the existing synthetic data generator under `artifacts/synthetic/*`.

## Current Repo Facts
- Synthetic dataset generation is already working through `scripts/data/collect_data.py`.
- The current collector emits consistent JSONL transitions and per-episode trees.
- The canonical runtime API lives in `env/`.
- The current RL file `rl/rainbow.py` is still a CartPole/CleanRL-based placeholder and is not usable for this project.
- The main train/eval entrypoints are under `scripts/train/` and `scripts/eval/`.

## Implementation Order

### Phase 1: Canonical runtime/data scaffolding
1. Add repo guidance in `AGENTS.md`.
2. Add canonical environment wrappers and helpers under `env/`:
   - `env/dsl_env.py`
   - `env/verifier.py`
   - `env/rewards.py`
   - `env/state_encoder.py`
3. Add dataset/replay loading under `data/`.
4. Validate with focused unit tests for schema, environment smoke, and dataset loading.

### Phase 2: RL core
1. Replace the existing `rl/rainbow.py` placeholder with a project-local masked Rainbow implementation.
2. Add:
   - `rl/categorical.py`
   - `rl/replay_buffer.py`
   - `rl/priorities.py`
3. Implement a dueling categorical Q-network in `models/q_network.py`.
4. Validate with unit tests for categorical projection, action masking, replay sampling, and output shapes.

### Phase 3: Training + evaluation scripts
1. Implement `scripts/train/train_rainbow.py` for:
   - offline bootstrap from `artifacts/synthetic/*/dataset.jsonl`
   - optional online fine-tuning against the canonical env
2. Implement `scripts/eval/eval_baselines.py` for:
   - random valid-action policy
   - heuristic policy
   - learned Rainbow policy
3. Add a deterministic `llm/repair.py` stub and wire a repair-proxy action in the canonical env.
4. Validate with a tiny offline-train smoke and baseline eval smoke.

## Non-negotiable Decisions
- The collector JSONL schema is the training contract; do not recollect data unless a real incompatibility appears.
- The canonical action table will include `REPAIR_FROM_FEEDBACK` even if offline data does not contain it yet.
- State encoding will be compact numeric features plus categorical one-hot blocks; no text model in the controller.
- Prioritized replay will use a simple proportional numpy-based sampler first, not a sum-tree.
- Logging will use standard Python logging plus TensorBoard; no external experiment platform is required.

## Validation Gates
- After env/data layer: run focused env + dataset tests.
- After RL core: run categorical/mask/replay/network tests.
- After train/eval scripts: run one tiny offline training smoke and one baseline eval smoke.

## Assumptions
- Python environment: `conda run -n hyperreasoning ...`
- Synthetic data lives under `artifacts/synthetic/<run_id>/`.
- Existing collection jobs may still be running; new code must not invalidate their on-disk outputs.
