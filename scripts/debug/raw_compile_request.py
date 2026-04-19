#!/usr/bin/env python3
"""Send one prepopulated compile prompt and print the raw HTTP response body."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import httpx
import orjson

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.dsl_env import SearchControlConfig, build_task_plan_bank
from env.dsl_schema import PlanDSL
from llm.compiler import build_edit_compiler_prompt
from llm.llm_client import LocalLLMClient
from llm.prompt_utils import load_task_context


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task-dir",
        type=Path,
        default=ROOT / "data/generated_tasks/hard/cache_invalidation_dependency_2073",
    )
    parser.add_argument("--llm-base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    task = load_task_context(args.task_dir)
    client = LocalLLMClient(base_url=args.llm_base_url)
    config = SearchControlConfig(proposal_source="heuristic", run_tests=False, max_verified_plans_per_task=1, seed=123)
    plan_bank = build_task_plan_bank(task, config, client=client)
    plan = PlanDSL.model_validate(plan_bank.entries[plan_bank.root_bank_ids[0]].plan)
    prompt = build_edit_compiler_prompt(task, plan)

    payload = {
        "model": client.model,
        "messages": [{"role": "user", "content": prompt + " Make sure to only include JSON in your response."}],
        "temperature": args.temperature,
        "cache_prompt": False,
    }
    if args.max_tokens is not None:
        payload["max_tokens"] = args.max_tokens

    response = httpx.post(
        args.llm_base_url.rstrip("/") + "/v1/chat/completions",
        content=orjson.dumps(payload),
        headers={"Content-Type": "application/json"},
        timeout=120.0,
    )
    response.raise_for_status()

    data = response.json()
    jout = data["choices"][0]["message"]["content"]
    # delete everything before <channel|>
    jout = jout.split("<channel|>")[-1]
    # remove ````json` and ` ````
    jout = jout.replace("```json", "").replace("```", "")
    print(jout)


    return 0


if __name__ == "__main__":
    raise SystemExit(main())
