#!/usr/bin/env python3
"""Run the standalone eval-set reporting pipeline from the scripts directory."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval.run_eval_pipeline import main


if __name__ == "__main__":
    raise SystemExit(main())
