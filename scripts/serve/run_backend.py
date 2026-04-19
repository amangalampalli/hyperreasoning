#!/usr/bin/env python3
"""Run the local FastAPI backend for the JetBrains plugin."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

import uvicorn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.api import create_app


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--llm-base-url", default="http://127.0.0.1:8080")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging()
    app = create_app(llm_base_url=args.llm_base_url)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
