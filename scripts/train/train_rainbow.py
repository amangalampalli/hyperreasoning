#!/usr/bin/env python3
"""Train a masked Rainbow agent from offline synthetic transitions, then optionally fine-tune online.

Supports resumable training by persisting:
- latest / best checkpoints
- replay buffer snapshot
- trainer state (step counters, patience state, EMA metrics, action counts)
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import logging
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.datasets import load_task_ids_from_manifest
from data.replay_dataset import EncodedTransition, encode_replay_dataset, load_offline_replay_dataset
from data.task_store import TaskStore
from env.dsl_env import ACTION_SPACE, DSLSearchEnv, SearchControlConfig
from env.state_encoder import StateEncoder
from rl.rainbow import RainbowAgent, RainbowConfig
from rl.replay_buffer import NStepEncodedStep, NStepTransitionAccumulator, PrioritizedReplayBuffer


LOGGER = logging.getLogger("train_rainbow")


class EMAMeter:
    """Simple exponential moving average tracker for noisy RL metrics."""

    def __init__(self, decay: float = 0.9) -> None:
        self.decay = decay
        self.value: float | None = None

    def update(self, raw: float) -> float:
        if self.value is None:
            self.value = raw
        else:
            self.value = self.decay * self.value + (1.0 - self.decay) * raw
        return self.value

    def state_dict(self) -> dict[str, float | None]:
        return {"decay": self.decay, "value": self.value}

    def load_state_dict(self, payload: dict[str, float | None]) -> None:
        self.decay = float(payload.get("decay", self.decay))
        self.value = payload.get("value")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dirs", nargs="+", default=[], help="Synthetic data run directories under artifacts/synthetic")
    parser.add_argument("--tasks-root", type=Path, default=ROOT / "data/generated_tasks")
    parser.add_argument("--task-manifest", type=Path, default=None, help="Optional manifest file listing tasks for online fine-tuning")
    parser.add_argument("--offline-task-manifest", type=Path, default=None, help="Optional manifest file listing tasks allowed for offline replay loading")
    parser.add_argument("--eval-task-manifest", type=Path, default=None, help="Optional held-out eval task manifest")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/models")
    parser.add_argument("--experiment-name", default="rainbow_v1")
    parser.add_argument("--resume-run-dir", type=Path, default=None, help="Resume training from an existing run directory")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--offline-updates", type=int, default=1000, help="Total offline updates target (resume-aware)")
    parser.add_argument("--online-episodes", type=int, default=0, help="Total online episodes target (resume-aware)")
    parser.add_argument("--max-online-steps", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--buffer-capacity", type=int, default=50000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--n-step", type=int, default=1)
    parser.add_argument("--num-atoms", type=int, default=51)
    parser.add_argument("--v-min", type=float, default=-2.0)
    parser.add_argument("--v-max", type=float, default=2.0)
    parser.add_argument("--target-update-interval", type=int, default=250)
    parser.add_argument("--priority-alpha", type=float, default=0.6)
    parser.add_argument("--priority-beta-start", type=float, default=0.4)
    parser.add_argument("--priority-beta-end", type=float, default=1.0)
    parser.add_argument("--epsilon-start", type=float, default=0.20)
    parser.add_argument("--epsilon-end", type=float, default=0.02)
    parser.add_argument("--llm-base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--run-tests", action="store_true")
    parser.add_argument("--allow-full-file-fallback", action="store_true")
    parser.add_argument("--max-verified-plans-per-task", type=int, default=1)
    parser.add_argument("--proposal-source", choices=["heuristic", "llm", "hybrid"], default="heuristic")
    parser.add_argument("--selection-epsilon", type=float, default=0.28)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=500, help="Run held-out evaluation every N update steps")
    parser.add_argument("--eval-num-tasks", type=int, default=10)
    parser.add_argument("--eval-episodes-per-task", type=int, default=1)
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument(
        "--early-stop-metric",
        choices=["mean_return", "visible_pass_rate", "compile_success_rate"],
        default="mean_return",
    )
    parser.add_argument("--ema-decay", type=float, default=0.9)
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def resolve_device(raw: str) -> torch.device:
    if raw == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(raw)


def linear_schedule(start: float, end: float, progress: float) -> float:
    progress = min(max(progress, 0.0), 1.0)
    return start + (end - start) * progress


def fit_encoder_from_transitions(transitions) -> StateEncoder:
    states: list[dict] = []
    for transition in transitions:
        states.append(transition.obs)
        if transition.next_obs is not None:
            states.append(transition.next_obs)
    return StateEncoder().fit(states)


def make_online_transition(
    *,
    encoder: StateEncoder,
    obs: dict,
    action: int,
    reward: float,
    next_obs: dict | None,
    terminated: bool,
    truncated: bool,
    action_mask: np.ndarray,
    next_action_mask: np.ndarray,
    task_id: str,
    metadata: dict,
    gamma: float,
) -> EncodedTransition:
    feature_dim = encoder.feature_dim
    zero_next = np.zeros(feature_dim, dtype=np.float32)
    return EncodedTransition(
        obs=encoder.encode_state(obs),
        action=action,
        reward=reward,
        next_obs=zero_next if next_obs is None else encoder.encode_state(next_obs),
        terminated=terminated,
        truncated=truncated,
        action_mask=np.asarray(action_mask, dtype=np.bool_),
        next_action_mask=np.asarray(next_action_mask, dtype=np.bool_),
        discount=gamma,
        task_id=task_id,
        metadata=dict(metadata),
    )


def log_action_distribution(writer: SummaryWriter, prefix: str, counts: Counter[int], *, step: int) -> None:
    total = sum(counts.values())
    if total <= 0:
        return
    for action_id, count in sorted(counts.items()):
        writer.add_scalar(f"{prefix}/action_count/{ACTION_SPACE[action_id]}", count, step)
        writer.add_scalar(f"{prefix}/action_fraction/{ACTION_SPACE[action_id]}", count / total, step)


def evaluate_learned_policy(
    *,
    agent: RainbowAgent,
    encoder: StateEncoder,
    task_store: TaskStore,
    llm_base_url: str,
    env_config: SearchControlConfig,
    episodes_per_task: int,
    seed: int,
    desc: str = "Held-out Eval",
) -> dict[str, float]:
    env = DSLSearchEnv(task_store=task_store, config=env_config, llm_base_url=llm_base_url, seed=seed)
    total_reward = 0.0
    compile_successes = 0
    visible_passes = 0
    lengths: list[int] = []
    episodes = 0
    compile_errors: Counter[str] = Counter()
    compile_attempts = 0
    total_steps = 0
    visible_test_failures: Counter[str] = Counter()
    tasks = list(env.task_store.iter_contexts())
    total_eval_episodes = len(tasks) * episodes_per_task
    with tqdm(total=total_eval_episodes, desc=desc, unit="ep", leave=False) as eval_bar:
        for task in tasks:
            for _ in range(episodes_per_task):
                obs, info = env.reset(task=task)
                done = False
                while not done:
                    action = agent.act(encoder.encode_state(obs), info["action_mask"], epsilon=0.0)
                    obs, reward, terminated, truncated, info = env.step(action)
                    total_reward += reward
                    if "compile_success" in info or "compile_error" in info:
                        compile_attempts += 1
                    compile_error = info.get("compile_error")
                    if compile_error:
                        compile_errors[compile_error] += 1
                    if info.get("compile_success") is True and info.get("visible_test_passed") is False:
                        stderr = (info.get("visible_test_stderr") or "").strip()
                        stdout = (info.get("visible_test_stdout") or "").strip()
                        summary = stderr or stdout or f"returncode={info.get('visible_test_returncode')}"
                        visible_test_failures[summary] += 1
                    done = terminated or truncated
                    total_steps += 1
            stats = env.episode_stats()
            compile_successes += stats.compile_successes
            visible_passes += stats.visible_test_passes
            lengths.append(stats.steps)
            episodes += 1
            eval_bar.update(1)
            eval_bar.set_postfix(
                compile=f"{compile_successes}/{episodes}",
                visible=f"{visible_passes}/{episodes}",
            )
    return {
        "episodes": episodes,
        "total_steps": total_steps,
        "compile_successes": compile_successes,
        "visible_passes": visible_passes,
        "mean_return": total_reward / max(1, episodes),
        "mean_episode_length": sum(lengths) / max(1, episodes),
        "compile_success_rate": compile_successes / max(1, episodes),
        "visible_pass_rate": visible_passes / max(1, episodes),
        "compile_attempts": compile_attempts,
        "top_compile_errors": dict(compile_errors.most_common(5)),
        "top_visible_test_failures": dict(visible_test_failures.most_common(5)),
    }


def save_named_checkpoint(agent: RainbowAgent, encoder: StateEncoder, path: Path, metadata: dict[str, object]) -> None:
    agent.save(path, encoder_state=encoder.to_dict())
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _counter_to_dict(counter: Counter[int]) -> dict[str, int]:
    return {str(key): int(value) for key, value in counter.items()}


def _counter_from_dict(payload: dict[str, int] | None) -> Counter[int]:
    counter: Counter[int] = Counter()
    for key, value in (payload or {}).items():
        counter[int(key)] = int(value)
    return counter


def save_training_state(
    *,
    path: Path,
    completed_offline_steps: int,
    completed_online_episodes: int,
    gradient_step: int,
    best_eval_metric: float,
    evals_without_improvement: int,
    ema_loss: EMAMeter,
    ema_q: EMAMeter,
    ema_reward: EMAMeter,
    offline_action_counter: Counter[int],
    online_action_counter: Counter[int],
    args: argparse.Namespace,
) -> None:
    payload = {
        "completed_offline_steps": completed_offline_steps,
        "completed_online_episodes": completed_online_episodes,
        "gradient_step": gradient_step,
        "best_eval_metric": best_eval_metric,
        "evals_without_improvement": evals_without_improvement,
        "ema_loss": ema_loss.state_dict(),
        "ema_q": ema_q.state_dict(),
        "ema_reward": ema_reward.state_dict(),
        "offline_action_counter": _counter_to_dict(offline_action_counter),
        "online_action_counter": _counter_to_dict(online_action_counter),
        "args": vars(args),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def load_training_state(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def append_metrics_event(path: Path, event: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True, default=str) + "\n")


def main() -> int:
    args = parse_args()
    setup_logging()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)

    resume = args.resume_run_dir is not None
    run_dirs = [Path(path) for path in args.run_dirs]
    if not run_dirs and not resume and args.online_episodes <= 0:
        raise SystemExit("Provide at least one --run-dirs source or set --online-episodes > 0")

    training_state_path: Path
    replay_path: Path
    latest_checkpoint_path: Path
    best_checkpoint_path: Path
    checkpoint_path: Path
    encoder_path: Path
    config_path: Path
    metrics_path: Path
    completed_offline_steps = 0
    completed_online_episodes = 0
    gradient_step = 0
    best_eval_metric = float("-inf")
    evals_without_improvement = 0
    ema_loss = EMAMeter(args.ema_decay)
    ema_q = EMAMeter(args.ema_decay)
    ema_reward = EMAMeter(args.ema_decay)
    offline_action_counter: Counter[int] = Counter()
    online_action_counter: Counter[int] = Counter()

    if resume:
        run_dir = args.resume_run_dir.resolve()
        training_state_path = run_dir / "trainer_state.json"
        replay_path = run_dir / "replay_buffer.npz"
        latest_checkpoint_path = run_dir / "latest.pt"
        best_checkpoint_path = run_dir / "best.pt"
        checkpoint_path = run_dir / "checkpoint.pt"
        encoder_path = run_dir / "state_encoder.json"
        config_path = run_dir / "train_config.json"
        metrics_path = run_dir / "metrics.jsonl"
        if not latest_checkpoint_path.exists() or not replay_path.exists() or not training_state_path.exists():
            raise SystemExit("Resume requires latest.pt, replay_buffer.npz, and trainer_state.json")
        agent, encoder_state = RainbowAgent.load(latest_checkpoint_path, device=device)
        if encoder_state is None:
            raise SystemExit("Resume checkpoint is missing encoder state")
        encoder = StateEncoder.from_dict(encoder_state)
        replay = PrioritizedReplayBuffer.load(replay_path)
        trainer_state = load_training_state(training_state_path)
        completed_offline_steps = int(trainer_state.get("completed_offline_steps", 0))
        completed_online_episodes = int(trainer_state.get("completed_online_episodes", 0))
        gradient_step = int(trainer_state.get("gradient_step", agent.train_steps))
        best_eval_metric = float(trainer_state.get("best_eval_metric", float("-inf")))
        evals_without_improvement = int(trainer_state.get("evals_without_improvement", 0))
        ema_loss.load_state_dict(dict(trainer_state.get("ema_loss", {})))
        ema_q.load_state_dict(dict(trainer_state.get("ema_q", {})))
        ema_reward.load_state_dict(dict(trainer_state.get("ema_reward", {})))
        offline_action_counter = _counter_from_dict(trainer_state.get("offline_action_counter"))
        online_action_counter = _counter_from_dict(trainer_state.get("online_action_counter"))
        LOGGER.info(
            "resuming from %s (offline_steps=%s online_episodes=%s gradient_step=%s replay=%s)",
            run_dir,
            completed_offline_steps,
            completed_online_episodes,
            gradient_step,
            len(replay),
        )
    else:
        offline_task_ids = load_task_ids_from_manifest(args.offline_task_manifest) if args.offline_task_manifest is not None else None
        offline_transitions = (
            load_offline_replay_dataset(run_dirs, gamma=args.gamma, n_step=args.n_step, allowed_task_ids=offline_task_ids)
            if run_dirs
            else []
        )
        if offline_task_ids is not None:
            LOGGER.info(
                "loaded offline train manifest %s with %s task ids; kept %s transitions",
                args.offline_task_manifest,
                len(offline_task_ids),
                len(offline_transitions),
            )
        encoder = fit_encoder_from_transitions(offline_transitions) if offline_transitions else StateEncoder().fit([])
        encoded_offline = encode_replay_dataset(offline_transitions, encoder) if offline_transitions else []
        buffer_capacity = max(args.buffer_capacity, len(encoded_offline) + max(1, args.online_episodes * args.max_online_steps))
        replay = PrioritizedReplayBuffer(
            capacity=buffer_capacity,
            obs_dim=max(1, encoder.feature_dim),
            num_actions=len(encoded_offline[0].action_mask) if encoded_offline else len(ACTION_SPACE),
            alpha=args.priority_alpha,
        )
        replay.extend(encoded_offline)
        agent = RainbowAgent(
            RainbowConfig(
                input_dim=max(1, encoder.feature_dim),
                num_actions=replay.action_masks.shape[1],
                num_atoms=args.num_atoms,
                v_min=args.v_min,
                v_max=args.v_max,
                learning_rate=args.learning_rate,
                gamma=args.gamma,
                target_update_interval=args.target_update_interval,
                batch_size=args.batch_size,
            ),
            device=device,
        )
        timestamp = int(time.time())
        run_name = f"{args.experiment_name}_{timestamp}"
        run_dir = args.output_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        training_state_path = run_dir / "trainer_state.json"
        replay_path = run_dir / "replay_buffer.npz"
        latest_checkpoint_path = run_dir / "latest.pt"
        best_checkpoint_path = run_dir / "best.pt"
        checkpoint_path = run_dir / "checkpoint.pt"
        encoder_path = run_dir / "state_encoder.json"
        config_path = run_dir / "train_config.json"
        metrics_path = run_dir / "metrics.jsonl"

    writer = SummaryWriter(log_dir=str(run_dir / "tb"))
    eval_task_store = None
    if args.eval_task_manifest is not None:
        eval_task_store = TaskStore.from_manifest(args.eval_task_manifest, limit=args.eval_num_tasks)
        LOGGER.info("loaded held-out eval task manifest %s with %s tasks", args.eval_task_manifest, len(eval_task_store))

    def persist_state(*, latest_metadata: dict[str, object] | None = None) -> None:
        if latest_metadata is not None:
            save_named_checkpoint(agent, encoder, latest_checkpoint_path, latest_metadata)
        replay.save(replay_path)
        save_training_state(
            path=training_state_path,
            completed_offline_steps=completed_offline_steps,
            completed_online_episodes=completed_online_episodes,
            gradient_step=gradient_step,
            best_eval_metric=best_eval_metric,
            evals_without_improvement=evals_without_improvement,
            ema_loss=ema_loss,
            ema_q=ema_q,
            ema_reward=ema_reward,
            offline_action_counter=offline_action_counter,
            online_action_counter=online_action_counter,
            args=args,
        )

    total_updates = max(1, args.offline_updates + args.online_episodes * args.max_online_steps)

    try:
        with tqdm(total=args.offline_updates, initial=completed_offline_steps, desc="Offline Train", unit="step") as offline_bar:
            for step in range(completed_offline_steps, args.offline_updates):
                beta = linear_schedule(args.priority_beta_start, args.priority_beta_end, step / max(1, total_updates - 1))
                batch = replay.sample(args.batch_size, beta=beta)
                metrics, priorities = agent.update(batch)
                replay.update_priorities(batch.indices, priorities)
                offline_action_counter.update(int(action) for action in batch.actions.tolist())
                loss_ema = ema_loss.update(metrics.loss)
                q_ema = ema_q.update(metrics.mean_q)
                reward_ema = ema_reward.update(metrics.mean_reward)
                completed_offline_steps = step + 1
                gradient_step = max(gradient_step, completed_offline_steps)
                offline_bar.update(1)
                offline_bar.set_postfix(loss=f"{loss_ema:.4f}", q=f"{q_ema:.4f}", reward=f"{reward_ema:.4f}")

                if step % args.log_interval == 0 or step == args.offline_updates - 1:
                    LOGGER.info(
                        "offline step=%s loss=%.4f ema_loss=%.4f mean_q=%.4f ema_q=%.4f mean_reward=%.4f ema_reward=%.4f buffer=%s",
                        step,
                        metrics.loss,
                        loss_ema,
                        metrics.mean_q,
                        q_ema,
                        metrics.mean_reward,
                        reward_ema,
                        len(replay),
                    )
                    writer.add_scalar("offline/loss", metrics.loss, step)
                    writer.add_scalar("offline/loss_ema", loss_ema, step)
                    writer.add_scalar("offline/mean_q", metrics.mean_q, step)
                    writer.add_scalar("offline/mean_q_ema", q_ema, step)
                    writer.add_scalar("offline/mean_reward", metrics.mean_reward, step)
                    writer.add_scalar("offline/mean_reward_ema", reward_ema, step)
                    writer.add_scalar("offline/mean_priority", metrics.mean_priority, step)
                    log_action_distribution(writer, "offline", offline_action_counter, step=step)
                    append_metrics_event(
                        metrics_path,
                        {
                            "event": "offline_train",
                            "step": step,
                            "loss": metrics.loss,
                            "loss_ema": loss_ema,
                            "mean_q": metrics.mean_q,
                            "mean_q_ema": q_ema,
                            "mean_reward": metrics.mean_reward,
                            "mean_reward_ema": reward_ema,
                            "mean_priority": metrics.mean_priority,
                            "buffer_size": len(replay),
                        },
                    )

                if eval_task_store is not None and ((step + 1) % args.eval_every == 0 or step == args.offline_updates - 1):
                    eval_metrics = evaluate_learned_policy(
                        agent=agent,
                        encoder=encoder,
                        task_store=eval_task_store,
                        llm_base_url=args.llm_base_url,
                        env_config=SearchControlConfig(
                            proposal_source=args.proposal_source,
                            run_tests=args.run_tests,
                            allow_full_file_fallback=args.allow_full_file_fallback,
                            max_verified_plans_per_task=args.max_verified_plans_per_task,
                            selection_epsilon=args.selection_epsilon,
                            seed=args.seed,
                        ),
                        episodes_per_task=args.eval_episodes_per_task,
                        seed=args.seed,
                    )
                    metric_value = float(eval_metrics[args.early_stop_metric])
                    writer.add_scalar("eval/mean_return", eval_metrics["mean_return"], step)
                    writer.add_scalar("eval/mean_episode_length", eval_metrics["mean_episode_length"], step)
                    writer.add_scalar("eval/compile_success_rate", eval_metrics["compile_success_rate"], step)
                    writer.add_scalar("eval/visible_pass_rate", eval_metrics["visible_pass_rate"], step)
                    writer.add_scalar("eval/compile_successes", eval_metrics["compile_successes"], step)
                    writer.add_scalar("eval/visible_passes", eval_metrics["visible_passes"], step)
                    writer.add_scalar("eval/episodes", eval_metrics["episodes"], step)
                    writer.add_scalar("eval/compile_attempts", eval_metrics["compile_attempts"], step)
                    LOGGER.info(
                        "eval step=%s mean_return=%.4f compile_success=%s/%s visible_pass=%s/%s compile_attempts=%s mean_episode_length=%.4f",
                        step,
                        eval_metrics["mean_return"],
                        eval_metrics["compile_successes"],
                        eval_metrics["episodes"],
                        eval_metrics["visible_passes"],
                        eval_metrics["episodes"],
                        eval_metrics["compile_attempts"],
                        eval_metrics["mean_episode_length"],
                    )
                    if eval_metrics["top_compile_errors"]:
                        LOGGER.warning("eval compile errors: %s", eval_metrics["top_compile_errors"])
                    if eval_metrics["top_visible_test_failures"]:
                        LOGGER.warning("eval visible test failures: %s", eval_metrics["top_visible_test_failures"])
                    append_metrics_event(
                        metrics_path,
                        {
                            "event": "eval",
                            "step": step,
                            **eval_metrics,
                        },
                    )
                    persist_state(latest_metadata={"step": step, "type": "latest", "eval_metrics": eval_metrics})
                    if metric_value > best_eval_metric:
                        best_eval_metric = metric_value
                        evals_without_improvement = 0
                        save_named_checkpoint(
                            agent,
                            encoder,
                            best_checkpoint_path,
                            {"step": step, "type": "best", "metric": args.early_stop_metric, "eval_metrics": eval_metrics},
                        )
                    else:
                        evals_without_improvement += 1
                        if args.early_stop_patience > 0 and evals_without_improvement >= args.early_stop_patience:
                            LOGGER.info(
                                "early stopping triggered after %s evals without improving %s",
                                evals_without_improvement,
                                args.early_stop_metric,
                            )
                            break

        if args.online_episodes > 0:
            env_config = SearchControlConfig(
                max_steps_per_episode=args.max_online_steps,
                proposal_source=args.proposal_source,
                run_tests=args.run_tests,
                allow_full_file_fallback=args.allow_full_file_fallback,
                max_verified_plans_per_task=args.max_verified_plans_per_task,
                selection_epsilon=args.selection_epsilon,
                seed=args.seed,
            )
            task_store = TaskStore.from_manifest(args.task_manifest) if args.task_manifest is not None else TaskStore.from_tasks_root(args.tasks_root)
            env = DSLSearchEnv(task_store=task_store, config=env_config, llm_base_url=args.llm_base_url, seed=args.seed)
            accumulator = NStepTransitionAccumulator(n_step=args.n_step, gamma=args.gamma)
            with tqdm(total=args.online_episodes, initial=completed_online_episodes, desc="Online Fine-tune", unit="ep") as online_bar:
                for episode_index in range(completed_online_episodes, args.online_episodes):
                    obs, info = env.reset()
                    done = False
                    episode_return = 0.0
                    while not done:
                        progress = gradient_step / max(1, total_updates - 1)
                        epsilon = linear_schedule(args.epsilon_start, args.epsilon_end, progress)
                        action = agent.act(encoder.encode_state(obs), info["action_mask"], epsilon=epsilon)
                        next_obs, reward, terminated, truncated, step_info = env.step(action)
                        next_mask = step_info["action_mask"] if next_obs is not None else np.zeros_like(info["action_mask"], dtype=np.bool_)

                        if args.n_step <= 1:
                            replay.add(
                                make_online_transition(
                                    encoder=encoder,
                                    obs=obs,
                                    action=action,
                                    reward=reward,
                                    next_obs=next_obs,
                                    terminated=terminated,
                                    truncated=truncated,
                                    action_mask=info["action_mask"],
                                    next_action_mask=next_mask,
                                    task_id=info["task_id"],
                                    metadata=step_info,
                                    gamma=args.gamma,
                                )
                            )
                            online_action_counter[action] += 1
                        else:
                            step_item = NStepEncodedStep(
                                obs=encoder.encode_state(obs),
                                action=action,
                                reward=reward,
                                next_obs=np.zeros(encoder.feature_dim, dtype=np.float32) if next_obs is None else encoder.encode_state(next_obs),
                                terminated=terminated,
                                truncated=truncated,
                                action_mask=np.asarray(info["action_mask"], dtype=np.bool_),
                                next_action_mask=np.asarray(next_mask, dtype=np.bool_),
                                task_id=info["task_id"],
                                metadata=step_info,
                            )
                            for transition in accumulator.push(step_item):
                                replay.add(transition)
                            online_action_counter[action] += 1

                        if len(replay) >= args.batch_size:
                            beta = linear_schedule(args.priority_beta_start, args.priority_beta_end, gradient_step / max(1, total_updates - 1))
                            batch = replay.sample(args.batch_size, beta=beta)
                            metrics, priorities = agent.update(batch)
                            replay.update_priorities(batch.indices, priorities)
                            if gradient_step % args.log_interval == 0:
                                writer.add_scalar("online/loss", metrics.loss, gradient_step)
                                writer.add_scalar("online/mean_q", metrics.mean_q, gradient_step)
                                writer.add_scalar("online/mean_reward", metrics.mean_reward, gradient_step)
                                writer.add_scalar("online/mean_priority", metrics.mean_priority, gradient_step)
                                log_action_distribution(writer, "online", online_action_counter, step=gradient_step)
                            gradient_step += 1

                        episode_return += reward
                        done = terminated or truncated
                        obs, info = next_obs, step_info

                    stats = env.episode_stats()
                    completed_online_episodes = episode_index + 1
                    online_bar.update(1)
                    online_bar.set_postfix(reward=f"{episode_return:.3f}", steps=stats.steps)
                    LOGGER.info(
                        "online episode=%s reward=%.4f steps=%s compile_successes=%s visible_passes=%s",
                        episode_index,
                        episode_return,
                        stats.steps,
                        stats.compile_successes,
                        stats.visible_test_passes,
                    )
                    writer.add_scalar("online/episode_return", episode_return, episode_index)
                    writer.add_scalar("online/episode_steps", stats.steps, episode_index)
                    append_metrics_event(
                        metrics_path,
                        {
                            "event": "online_episode",
                            "episode": episode_index,
                            "reward": episode_return,
                            "steps": stats.steps,
                            "compile_successes": stats.compile_successes,
                            "visible_passes": stats.visible_test_passes,
                        },
                    )
                    persist_state(latest_metadata={"step": gradient_step, "type": "latest_online", "episode_index": episode_index})

    except KeyboardInterrupt:
        LOGGER.info("training interrupted; saving resumable state to %s", run_dir)
        persist_state(latest_metadata={"step": gradient_step, "type": "interrupted"})
        writer.close()
        return 130

    encoder.save(encoder_path)
    agent.save(checkpoint_path, encoder_state=encoder.to_dict())
    save_named_checkpoint(agent, encoder, latest_checkpoint_path, {"step": gradient_step, "type": "latest_final"})
    replay.save(replay_path)
    save_training_state(
        path=training_state_path,
        completed_offline_steps=completed_offline_steps,
        completed_online_episodes=completed_online_episodes,
        gradient_step=gradient_step,
        best_eval_metric=best_eval_metric,
        evals_without_improvement=evals_without_improvement,
        ema_loss=ema_loss,
        ema_q=ema_q,
        ema_reward=ema_reward,
        offline_action_counter=offline_action_counter,
        online_action_counter=online_action_counter,
        args=args,
    )
    config_path.write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")
    writer.close()
    LOGGER.info("saved checkpoint to %s", checkpoint_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
