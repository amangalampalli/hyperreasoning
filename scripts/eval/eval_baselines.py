#!/usr/bin/env python3
"""Evaluate random, heuristic, and learned policies on the canonical DSL search environment."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import logging
from pathlib import Path
import random
import sys

import numpy as np
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.task_store import TaskStore
from env.dsl_env import DSLSearchEnv, SearchControlConfig
from env.state_encoder import StateEncoder
from rl.rainbow import RainbowAgent


LOGGER = logging.getLogger("eval_baselines")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-root", type=Path, default=ROOT / "data/generated_tasks")
    parser.add_argument("--task-manifest", type=Path, default=None, help="Optional manifest file listing held-out eval tasks")
    parser.add_argument("--num-tasks", type=int, default=10)
    parser.add_argument("--episodes-per-task", type=int, default=1)
    parser.add_argument("--policy", choices=["random", "heuristic", "rainbow", "all"], default="all")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--llm-base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--run-tests", action="store_true")
    parser.add_argument("--allow-full-file-fallback", action="store_true")
    parser.add_argument("--max-verified-plans-per-task", type=int, default=1)
    parser.add_argument("--proposal-source", choices=["heuristic", "llm", "hybrid"], default="heuristic")
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def choose_random_action(mask: np.ndarray, rng: random.Random) -> int:
    valid_indices = np.flatnonzero(mask)
    return int(rng.choice(valid_indices.tolist()))


def evaluate_policy(
    name: str,
    env: DSLSearchEnv,
    *,
    encoder: StateEncoder | None,
    agent: RainbowAgent | None,
    episodes_per_task: int,
) -> dict[str, float]:
    rng = random.Random(123)
    total_reward = 0.0
    compile_successes = 0
    visible_passes = 0
    episode_lengths: list[int] = []
    compile_errors: Counter[str] = Counter()
    compile_attempts = 0
    total_steps = 0
    visible_test_failures: Counter[str] = Counter()

    tasks = list(env.task_store.iter_contexts())
    total_episodes = len(tasks) * episodes_per_task
    with tqdm(total=total_episodes, desc=f"Eval {name}", unit="ep") as eval_bar:
        for task in tasks:
            for _ in range(episodes_per_task):
                obs, info = env.reset(task=task)
                done = False
                while not done:
                    if name == "random":
                        action = choose_random_action(info["action_mask"], rng)
                    elif name == "heuristic":
                        action = env.heuristic_action()
                    elif name == "rainbow":
                        if agent is None or encoder is None:
                            raise ValueError("Rainbow evaluation requires agent and encoder")
                        action = agent.act(encoder.encode_state(obs), info["action_mask"], epsilon=0.0)
                    else:
                        raise ValueError(f"Unsupported policy {name}")
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
            episode_lengths.append(stats.steps)
            eval_bar.update(1)
            eval_bar.set_postfix(
                compile=f"{compile_successes}/{len(episode_lengths)}",
                visible=f"{visible_passes}/{len(episode_lengths)}",
            )

    episodes = len(episode_lengths)
    return {
        "policy": name,
        "episodes": episodes,
        "total_steps": total_steps,
        "compile_successes": compile_successes,
        "visible_passes": visible_passes,
        "mean_return": total_reward / max(1, episodes),
        "mean_episode_length": sum(episode_lengths) / max(1, episodes),
        "compile_success_rate": compile_successes / max(1, episodes),
        "visible_pass_rate": visible_passes / max(1, episodes),
        "compile_attempts": compile_attempts,
        "top_compile_errors": dict(compile_errors.most_common(5)),
        "top_visible_test_failures": dict(visible_test_failures.most_common(5)),
    }


def main() -> int:
    args = parse_args()
    setup_logging()
    task_store = (
        TaskStore.from_manifest(args.task_manifest, limit=args.num_tasks)
        if args.task_manifest is not None
        else TaskStore.from_tasks_root(args.tasks_root, limit=args.num_tasks)
    )
    env = DSLSearchEnv(
        task_store=task_store,
        config=SearchControlConfig(
            proposal_source=args.proposal_source,
            run_tests=args.run_tests,
            allow_full_file_fallback=args.allow_full_file_fallback,
            max_verified_plans_per_task=args.max_verified_plans_per_task,
        ),
        llm_base_url=args.llm_base_url,
    )

    encoder = None
    agent = None
    if args.policy in {"rainbow", "all"}:
        if args.checkpoint is None:
            raise SystemExit("--checkpoint is required for rainbow evaluation")
        agent, encoder_state = RainbowAgent.load(args.checkpoint)
        if encoder_state is None:
            raise SystemExit("Checkpoint does not contain encoder state")
        encoder = StateEncoder.from_dict(encoder_state)

    policies = ["random", "heuristic", "rainbow"] if args.policy == "all" else [args.policy]
    results = [
        evaluate_policy(
            policy,
            env,
            encoder=encoder,
            agent=agent,
            episodes_per_task=args.episodes_per_task,
        )
        for policy in policies
    ]
    for result in results:
        LOGGER.info(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
