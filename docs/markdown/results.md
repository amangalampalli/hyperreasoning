# Eval Results

Source eval manifest: `data/splits/eval_10.txt`
Rainbow checkpoint: `artifacts/models/rainbow_offline_v1_1776237078/best.pt`
Paired units: `10` eval tasks

## Summary

| Method | Solve rate | Mean tests passed | Mean time | Mean tokens | Tokens / solved | Time / solved |
| --- | --- | --- | --- | --- | --- | --- |
| Rainbow | 80.0% | 80.0% | 15,353 ms | 1,694 | 2,118 | 19,192 ms |
| Heuristic | 40.0% | 40.0% | 23,144 ms | 2,340 | 5,849 | 57,859 ms |
| Random | 30.0% | 37.5% | 23,009 ms | 2,342 | 7,807 | 76,698 ms |
| One-shot | 30.0% | 37.5% | 31,016 ms | 3,649 | 12,164 | 103,386 ms |

## Interpretation

`Rainbow` is the top method on the held-out eval split by solve rate, with the paired comparisons below computed task-by-task against Rainbow.

## Paired Comparisons

- Rainbow improves solve rate by +0.40 (95% CI [0.10, 0.70], p=0.125) vs heuristic.
- Rainbow improves solve rate by +0.50 (95% CI [0.20, 0.80], p=0.062) vs random.
- Rainbow improves solve rate by +0.50 (95% CI [0.20, 0.80], p=0.062) vs one shot.
- Rainbow improves mean fraction of tests passed by +0.40 (95% CI [0.17, 0.65], p=0.031) vs heuristic.
- Rainbow reduces elapsed time by 7,790 ms (95% CI [2,890, 13,756] ms, p=0.004) vs heuristic.
- Rainbow reduces total tokens by 645 (95% CI [293, 1,085], p=0.002) vs heuristic.
- Rainbow improves mean fraction of tests passed by +0.42 (95% CI [0.20, 0.65], p=0.016) vs random.
- Rainbow reduces elapsed time by 7,656 ms (95% CI [2,729, 13,435] ms, p=0.008) vs random.
- Rainbow reduces total tokens by 648 (95% CI [299, 1,083], p=0.002) vs random.
- Rainbow improves mean fraction of tests passed by +0.42 (95% CI [0.20, 0.65], p=0.016) vs one shot.
- Rainbow reduces elapsed time by 15,662 ms (95% CI [8,976, 22,967] ms, p=0.002) vs one shot.
- Rainbow reduces total tokens by 1,955 (95% CI [1,443, 2,558], p=0.002) vs one shot.

## Graphs

- [Solve rate](../png/solve_rate.png)
- [Fraction tests passed](../png/fraction_tests_passed.png)
- [Time per task](../png/time_per_task.png)
- [Tokens per task](../png/tokens_per_task.png)
- [Branches explored per task](../png/branches_per_task.png)
- [Rainbow vs heuristic](../png/rainbow_vs_heuristic_delta.png)
- [Rainbow vs random](../png/rainbow_vs_random_delta.png)
- [Rainbow vs one-shot](../png/rainbow_vs_one_shot_delta.png)

## Assumptions

- `one_shot` is the exported name for the existing internal `oneshot` policy.
- `tests_passed` and `tests_total` are measured from each task's visible eval tests.
- Paired confidence intervals use 10,000 bootstrap resamples.
- P-values use exact paired tests: McNemar for `solved`, and exact sign-flip randomization tests for the continuous paired deltas.
