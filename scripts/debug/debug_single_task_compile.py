#!/usr/bin/env python3
"""Debug one task's compile request against the local llama.cpp server."""

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
from llm.compiler import CompilePlanError, build_edit_compiler_prompt, compile_plan_to_code
from llm.llm_client import LocalLLMClient
from llm.prompt_utils import load_task_context


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task-dir",
        type=Path,
        default=ROOT / "data/generated_tasks/hard/cache_invalidation_dependency_2073",
        help="Task directory to inspect.",
    )
    parser.add_argument(
        "--llm-base-url",
        default="http://127.0.0.1:8080",
        help="Local llama.cpp server URL.",
    )
    parser.add_argument(
        "--proposal-source",
        choices=["heuristic", "llm", "hybrid"],
        default="heuristic",
        help="Plan-bank root generation source.",
    )
    parser.add_argument(
        "--plan-index",
        type=int,
        default=0,
        help="Root plan index to inspect.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Compiler request temperature.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Optional max_tokens cap for the raw HTTP request. Omit to send no cap.",
    )
    parser.add_argument(
        "--prompt-out",
        type=Path,
        default=ROOT / "artifacts" / "debug_compile_prompt.txt",
        help="Where to save the exact prompt text.",
    )
    parser.add_argument(
        "--response-json",
        type=Path,
        default=ROOT / "artifacts" / "debug_compile_response.json",
        help="Where to save the full request/response capture JSON.",
    )
    parser.add_argument(
        "--response-body",
        type=Path,
        default=ROOT / "artifacts" / "debug_compile_response_body.txt",
        help="Where to save the exact raw HTTP response body text.",
    )
    return parser.parse_args()


def print_section(title: str, body: str) -> None:
    print(f"\n== {title} ==")
    print(body)


def main() -> int:
    args = parse_args()
    task = load_task_context(args.task_dir)
    client = LocalLLMClient(base_url=args.llm_base_url)
    config = SearchControlConfig(
        proposal_source=args.proposal_source,
        run_tests=False,
        max_verified_plans_per_task=1,
        seed=123,
    )
    plan_bank = build_task_plan_bank(task, config, client=client)
    if not plan_bank.root_bank_ids:
        raise SystemExit("No root plans were generated.")
    if args.plan_index < 0 or args.plan_index >= len(plan_bank.root_bank_ids):
        raise SystemExit(f"--plan-index must be in [0, {len(plan_bank.root_bank_ids) - 1}]")

    bank_id = plan_bank.root_bank_ids[args.plan_index]
    plan = PlanDSL.model_validate(plan_bank.entries[bank_id].plan)
    prompt = build_edit_compiler_prompt(task, plan)
    args.prompt_out.parent.mkdir(parents=True, exist_ok=True)
    args.prompt_out.write_text(prompt, encoding="utf-8")

    print_section("Task", f"{task.task_id} ({task.family})")
    print_section("Plan", orjson.dumps(plan.model_dump(), option=orjson.OPT_INDENT_2).decode("utf-8"))
    print_section("Prompt Preview", prompt[:2000])

    payload = {
        "model": client.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": args.temperature,
        "cache_prompt": False,
        "response_format": {"type": "json_object"},
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
    raw_body = response.text
    try:
        parsed = orjson.loads(response.content)
    except orjson.JSONDecodeError:
        parsed = None
    args.response_json.parent.mkdir(parents=True, exist_ok=True)
    args.response_body.parent.mkdir(parents=True, exist_ok=True)
    args.response_body.write_text(raw_body, encoding="utf-8")

    if isinstance(parsed, dict):
        message = parsed.get("choices", [{}])[0].get("message", {})
    else:
        message = {}
    content = message.get("content") or ""
    reasoning_content = message.get("reasoning_content") or ""
    capture_payload = {
        "task_id": task.task_id,
        "family": task.family,
        "task_dir": str(task.task_dir),
        "request_url": args.llm_base_url.rstrip("/") + "/v1/chat/completions",
        "request_payload": payload,
        "http_status": response.status_code,
        "response_headers": dict(response.headers),
        "raw_response_body": raw_body,
        "parsed_response": parsed,
        "message_content": content,
        "message_reasoning_content": reasoning_content,
    }
    args.response_json.write_text(
        orjson.dumps(capture_payload, option=orjson.OPT_INDENT_2).decode("utf-8"),
        encoding="utf-8",
    )

    print_section(
        "Response Lengths",
        f"content_len={len(content)}\nreasoning_len={len(reasoning_content)}\n"
        f"prompt={args.prompt_out}\ncapture_json={args.response_json}\nraw_body={args.response_body}",
    )
    print_section("Content (repr)", repr(content[:2000]))
    print_section("Reasoning Preview", reasoning_content[:4000] or "<empty>")

    try:
        compile_result = compile_plan_to_code(
            task,
            plan,
            temperature=args.temperature,
            client=client,
            return_debug=True,
            allow_full_file_fallback=False,
        )
        compiled_files = compile_result["compiled_files"]
        print_section("Compile Result", f"Compiled files: {sorted(compiled_files)}")
    except CompilePlanError as exc:
        print_section("Compile Error", str(exc))
        print_section("Compile Raw Response", exc.raw_response or "<empty>")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
