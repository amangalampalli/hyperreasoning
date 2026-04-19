#!/usr/bin/env python3
"""Collect cheap full-search-control trees for Rainbow training."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys
from typing import Any

import orjson
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from artifacts.io import make_run_id, save_json, save_jsonl
from env.dsl_env import SearchControlConfig, collect_task_dataset
from llm.llm_client import LocalLLMClient
from llm.prompt_utils import load_task_context
from data.task_manifest import discover_task_dirs, filter_task_dirs, is_task_completed, load_task_manifest, shuffle_task_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", type=Path, default=None, help="Optional single task directory")
    parser.add_argument("--manifest", type=Path, default=None, help="Optional manifest of task directories")
    parser.add_argument(
        "--tasks-root",
        type=Path,
        default=ROOT / "data/generated_tasks/hard",
        help="Root containing generated task directories",
    )
    parser.add_argument("--num-tasks", type=int, default=10, help="Number of tasks to process")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap after filtering/shuffling")
    parser.add_argument("--resume", action="store_true", help="Skip tasks already completed in the target run directory")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle task order deterministically")
    parser.add_argument("--seed", type=int, default=123, help="Base seed")
    parser.add_argument("--families", default=None, help="Comma-separated family filter")
    parser.add_argument("--difficulties", default=None, help="Comma-separated difficulty filter")
    parser.add_argument("--episodes-per-task", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--max-bank-depth", type=int, default=2)
    parser.add_argument("--root-candidate-batches", type=int, default=1)
    parser.add_argument("--root-candidates-per-batch", type=int, default=4)
    parser.add_argument("--max-root-plans", type=int, default=6)
    parser.add_argument("--refinement-branching", type=int, default=3)
    parser.add_argument("--select-child-slots", type=int, default=4)
    parser.add_argument("--initial-root-reveal", type=int, default=2)
    parser.add_argument("--initial-refine-reveal", type=int, default=2)
    parser.add_argument("--request-batch-size", type=int, default=2)
    parser.add_argument("--proposal-source", choices=["heuristic", "llm", "hybrid"], default="heuristic")
    parser.add_argument("--selection-epsilon", type=float, default=0.28)
    parser.add_argument("--run-tests", action="store_true")
    parser.add_argument("--max-verified-plans-per-task", type=int, default=8)
    parser.add_argument("--compiler-temp", type=float, default=0.2)
    parser.add_argument("--allow-full-file-fallback", action="store_true")
    parser.add_argument(
        "--max-llm-workers",
        type=int,
        default=1,
        help="Number of tasks to process concurrently; useful when the local LLM server handles parallel requests",
    )
    parser.add_argument("--llm-base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--timeout", type=float, default=12.0)
    parser.add_argument("--artifacts-dir", type=Path, default=ROOT / "artifacts/synthetic")
    parser.add_argument("--run-id", default=None)
    return parser.parse_args()


def _parse_filter_set(raw: str | None) -> set[str] | None:
    if raw is None:
        return None
    values = {item.strip() for item in raw.split(",") if item.strip()}
    return values or None


def resolve_task_dirs(
    *,
    task_dir: Path | None,
    tasks_root: Path,
    manifest: Path | None,
    num_tasks: int,
    families: set[str] | None,
    difficulties: set[str] | None,
    shuffle: bool,
    seed: int,
    limit: int | None,
) -> list[Path]:
    if task_dir is not None:
        return [task_dir]
    if manifest is not None:
        task_dirs = load_task_manifest(manifest)
    else:
        task_dirs = discover_task_dirs(tasks_root)
    task_dirs = filter_task_dirs(task_dirs, families=families, difficulties=difficulties)
    if shuffle:
        task_dirs = shuffle_task_dirs(task_dirs, seed=seed)
    if limit is not None:
        task_dirs = task_dirs[:limit]
    else:
        task_dirs = task_dirs[:num_tasks]
    return task_dirs


def _append_jsonl(path: Path, rows: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        for row in rows:
            payload = row.model_dump() if hasattr(row, "model_dump") else row
            handle.write(orjson.dumps(payload))
            handle.write(b"\n")


def _task_output_dir(run_dir: Path, task_id: str) -> Path:
    return run_dir / task_id


def _load_existing_summary(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = orjson.loads(path.read_bytes())
    except orjson.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _collect_run_task_summaries(run_dir: Path) -> list[dict[str, Any]]:
    task_summaries: list[dict[str, Any]] = []
    if not run_dir.exists():
        return task_summaries
    for child in sorted(run_dir.iterdir()):
        if not child.is_dir():
            continue
        summary = _load_existing_summary(child / "summary.json")
        if summary is not None:
            task_summaries.append(summary)
    return task_summaries


def _collect_one_task(
    *,
    task_dir: Path,
    config: SearchControlConfig,
    llm_base_url: str,
) -> tuple[dict[str, Any], list[Any]]:
    task = load_task_context(task_dir)
    client = LocalLLMClient(base_url=llm_base_url)
    bundle = collect_task_dataset(task, config, client=client)
    return {
        "task_id": task.task_id,
        "family": task.family,
        "bundle": bundle,
    }, bundle.transitions


def main() -> int:
    args = parse_args()
    if args.resume and not args.run_id:
        print("--resume requires --run-id so the collector knows which run to continue.")
        return 1

    run_id = args.run_id or make_run_id()
    families = _parse_filter_set(args.families)
    difficulties = _parse_filter_set(args.difficulties)
    config = SearchControlConfig(
        episodes_per_task=args.episodes_per_task,
        max_steps_per_episode=args.max_steps,
        max_bank_depth=args.max_bank_depth,
        root_candidate_batches=args.root_candidate_batches,
        root_candidates_per_batch=args.root_candidates_per_batch,
        max_root_plans=args.max_root_plans,
        refinement_branching=args.refinement_branching,
        select_child_slots=args.select_child_slots,
        initial_root_reveal=args.initial_root_reveal,
        initial_refine_reveal=args.initial_refine_reveal,
        request_batch_size=args.request_batch_size,
        proposal_source=args.proposal_source,
        selection_epsilon=args.selection_epsilon,
        compiler_temp=args.compiler_temp,
        run_tests=args.run_tests,
        python_bin=args.python_bin,
        timeout_s=args.timeout,
        allow_full_file_fallback=args.allow_full_file_fallback,
        max_verified_plans_per_task=args.max_verified_plans_per_task,
        seed=args.seed,
    )

    task_dirs = resolve_task_dirs(
        task_dir=args.task_dir,
        tasks_root=args.tasks_root,
        manifest=args.manifest,
        num_tasks=args.num_tasks,
        families=families,
        difficulties=difficulties,
        shuffle=args.shuffle,
        seed=args.seed,
        limit=args.limit,
    )
    if not task_dirs:
        print("No task directories found.")
        return 1

    run_dir = args.artifacts_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = run_dir / "dataset.jsonl"
    if dataset_path.exists() and not args.resume:
        dataset_path.unlink()

    tasks_skipped_resume = 0
    current_run_summaries: list[dict[str, Any]] = []
    with tqdm(total=len(task_dirs), desc="Tasks", unit="task") as task_bar:
        if args.max_llm_workers <= 1:
            for task_index, task_dir in enumerate(task_dirs):
                task = load_task_context(task_dir)
                if args.resume and is_task_completed(run_dir, task.task_id):
                    tasks_skipped_resume += 1
                    tqdm.write(f"=== Skipping completed task {task.task_id} ({task.family}) ===")
                    task_bar.update(1)
                    continue
                tqdm.write(f"=== Collecting control dataset for {task.task_id} ({task.family}) ===")
                task_config = config.model_copy(update={"seed": args.seed + task_index * 1000})
                bundle = collect_task_dataset(
                    task,
                    task_config,
                    client=LocalLLMClient(base_url=args.llm_base_url),
                )

                output_dir = _task_output_dir(run_dir, task.task_id)
                save_json(output_dir / "plan_bank.json", bundle.plan_bank)
                save_json(output_dir / "summary.json", bundle.summary)
                save_jsonl(output_dir / "transitions.jsonl", bundle.transitions)
                episodes_dir = output_dir / "episodes"
                with tqdm(
                    total=len(bundle.episodes),
                    desc=f"{task.task_id}",
                    unit="ep",
                    leave=False,
                ) as episode_bar:
                    for episode in bundle.episodes:
                        save_json(episodes_dir / f"{episode.episode_id}.json", episode)
                        episode_bar.update(1)
                _append_jsonl(dataset_path, bundle.transitions)

                current_run_summaries.append(bundle.summary)
                tqdm.write(
                    f"episodes={bundle.summary['episodes']} transitions={bundle.summary['transitions']} "
                    f"bank={bundle.summary['bank_total_plans']} verified={bundle.summary['verifier']['verified_plans']}"
                )
                task_bar.update(1)
        else:
            future_map: dict[Any, tuple[int, Path]] = {}
            with ThreadPoolExecutor(max_workers=args.max_llm_workers) as executor:
                for task_index, task_dir in enumerate(task_dirs):
                    if args.resume:
                        task = load_task_context(task_dir)
                        if is_task_completed(run_dir, task.task_id):
                            tasks_skipped_resume += 1
                            tqdm.write(f"=== Skipping completed task {task.task_id} ({task.family}) ===")
                            task_bar.update(1)
                            continue
                    task_config = config.model_copy(update={"seed": args.seed + task_index * 1000})
                    future = executor.submit(
                        _collect_one_task,
                        task_dir=task_dir,
                        config=task_config,
                        llm_base_url=args.llm_base_url,
                    )
                    future_map[future] = (task_index, task_dir)

                for future in as_completed(future_map):
                    _, task_dir = future_map[future]
                    result, transitions = future.result()
                    bundle = result["bundle"]
                    task_id = result["task_id"]
                    family = result["family"]
                    tqdm.write(f"=== Collected control dataset for {task_id} ({family}) ===")

                    output_dir = _task_output_dir(run_dir, task_id)
                    save_json(output_dir / "plan_bank.json", bundle.plan_bank)
                    save_json(output_dir / "summary.json", bundle.summary)
                    save_jsonl(output_dir / "transitions.jsonl", transitions)
                    episodes_dir = output_dir / "episodes"
                    for episode in bundle.episodes:
                        save_json(episodes_dir / f"{episode.episode_id}.json", episode)
                    _append_jsonl(dataset_path, transitions)

                    current_run_summaries.append(bundle.summary)
                    tqdm.write(
                        f"episodes={bundle.summary['episodes']} transitions={bundle.summary['transitions']} "
                        f"bank={bundle.summary['bank_total_plans']} verified={bundle.summary['verifier']['verified_plans']}"
                    )
                    task_bar.update(1)
                    task_bar.set_postfix(
                        task=task_dir.name,
                        transitions=sum(item["transitions"] for item in current_run_summaries),
                    )

    task_summaries = _collect_run_task_summaries(run_dir)
    run_summary = {
        "run_id": run_id,
        "config": config.model_dump(),
        "resume": args.resume,
        "tasks_processed": len(task_summaries),
        "tasks_processed_this_invocation": len(current_run_summaries),
        "tasks_skipped_resume": tasks_skipped_resume,
        "total_episodes": sum(item.get("episodes", 0) for item in task_summaries),
        "total_transitions": sum(item.get("transitions", 0) for item in task_summaries),
        "avg_bank_total_plans": (
            sum(item.get("bank_total_plans", 0) for item in task_summaries) / len(task_summaries)
            if task_summaries
            else 0.0
        ),
        "avg_verified_plans_per_task": (
            sum(item.get("verifier", {}).get("verified_plans", 0) for item in task_summaries) / len(task_summaries)
            if task_summaries
            else 0.0
        ),
        "per_task": task_summaries,
    }
    save_json(run_dir / "run_summary.json", run_summary)
    print(
        f"Saved {run_summary['total_episodes']} episodes and {run_summary['total_transitions']} transitions to {run_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
