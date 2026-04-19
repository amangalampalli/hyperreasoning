## PyCharm Demo Project

Source task: `data/generated_tasks/medium/stateful_iterator_resume_bug_1080`

Status: intentionally broken starting state for the live comparison demo.

Same-pipeline demo-folder scan:

- Rainbow: solved, 4/4 tests, 1311 tokens, 25.4s
- Heuristic: partial, 3/4 tests, 3306 tokens, 71.1s
- One-shot: solved, 4/4 tests, 4452 tokens, 82.4s
- Random: solved, 4/4 tests, 2989 tokens, 57.2s

Demo category:

- Secondary partial-quality edge
- Rainbow solved with fewer tokens/time

Family: resumable iterator state restore.

Entry point: `iterator_impl.py`

Target file: `iterator_impl.py`

Prompt:

`iterator_impl.py` contains a resumable iterator over grouped items. Fix
checkpoint and restore behavior so resuming never duplicates or skips values and
terminal state remains stable.
