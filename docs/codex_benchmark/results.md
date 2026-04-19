# Codex Benchmark Results

Rainbow source: `/Users/aditya/Developer/hyperreasoning/docs/raw_eval_results.jsonl`
Paired units: `10` eval tasks

## Summary

| Method | Solve rate | Mean tests passed | Mean time | Mean total tokens | Mean reasoning tokens | Mean execution tokens |
| --- | --- | --- | --- | --- | --- | --- |
| Rainbow | 80.0% | 80.0% | 15,353 ms | 1,694 | n/a | n/a |
| Codex 5.4 Low | 80.0% | 100.0% | 65,866 ms | 243,812 | n/a | n/a |
| Codex 5.4 Medium | 90.0% | 98.0% | 79,198 ms | 264,886 | n/a | n/a |
| Codex 5.4 High | 100.0% | 100.0% | 93,375 ms | 312,124 | n/a | n/a |

## Paired Comparisons

- Rainbow vs Codex 5.4 Low solve-rate delta: +0.00 (95% CI [-0.40, 0.40], p=1.000)
- Rainbow vs Codex 5.4 Low time delta: -50,513 ms (95% CI [-62009, -40680], p=0.002)
- Rainbow vs Codex 5.4 Low token delta: -242,118 (95% CI [-287760, -211537], p=0.002)
- Rainbow vs Codex 5.4 Medium solve-rate delta: -0.10 (95% CI [-0.40, 0.20], p=1.000)
- Rainbow vs Codex 5.4 Medium time delta: -63,845 ms (95% CI [-79251, -50453], p=0.002)
- Rainbow vs Codex 5.4 Medium token delta: -263,191 (95% CI [-299154, -226711], p=0.002)
- Rainbow vs Codex 5.4 High solve-rate delta: -0.20 (95% CI [-0.50, 0.00], p=0.500)
- Rainbow vs Codex 5.4 High time delta: -78,022 ms (95% CI [-95791, -62805], p=0.002)
- Rainbow vs Codex 5.4 High token delta: -310,429 (95% CI [-370630, -259492], p=0.002)

## Graphs

- [Solve rate](solve_rate.png)
- [Time per task](time_per_task.png)
- [Tokens per task](tokens_per_task.png)
- [Rainbow vs Codex 5.4 Low](rainbow_vs_codex_5_4_low_delta.png)
- [Rainbow vs Codex 5.4 Medium](rainbow_vs_codex_5_4_medium_delta.png)
- [Rainbow vs Codex 5.4 High](rainbow_vs_codex_5_4_high_delta.png)

## Assumptions

- Rainbow rows are loaded from the existing raw eval file and re-scored using all-tests-gated success (`tests_passed == tests_total`).
- Codex timing is measured around the full `codex exec` invocation.
- Reasoning/execution token splits are included only when Codex CLI exposes them cleanly; otherwise those fields stay blank and total tokens remain the primary metric.
