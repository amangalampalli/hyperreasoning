## Demo Projects

These folders are small PyCharm-loadable workspaces copied strictly from
`data/splits/eval_10.txt`.

## Recommended Demo Set

Use these when the demo needs to show Rainbow's current same-pipeline edge on
the exact demo workspaces.

Selection source: `.hyper/rainbow_edge_scans/demo_folder_records_latest.jsonl`.
This was recomputed directly from the demo folders, not from the tracked raw
eval artifact.

- [pycharm_eval_ast_transform_scope_bug_2058](/Users/aditya/Developer/hyperreasoning/demo/pycharm_eval_ast_transform_scope_bug_2058/README.md)
  - edge: Rainbow solved 4/4; one-shot failed 3/4
  - family: scope-aware AST rename
- [pycharm_eval_cache_invalidation_dependency_2073](/Users/aditya/Developer/hyperreasoning/demo/pycharm_eval_cache_invalidation_dependency_2073/README.md)
  - edge: same quality as competitors, materially fewer tokens/time
  - family: dependency invalidation cache
- [pycharm_eval_multi_file_interface_drift_4025](/Users/aditya/Developer/hyperreasoning/demo/pycharm_eval_multi_file_interface_drift_4025/README.md)
  - edge: same quality as competitors, materially fewer tokens/time
  - family: multi-file API/client contract drift
- [pycharm_eval_stateful_iterator_resume_bug_1080](/Users/aditya/Developer/hyperreasoning/demo/pycharm_eval_stateful_iterator_resume_bug_1080/README.md)
  - edge: Rainbow solved 4/4 with fewer tokens/time; heuristic only reached 3/4 tests
  - family: resumable iterator state restore
- [pycharm_eval_stateful_iterator_resume_bug_4007](/Users/aditya/Developer/hyperreasoning/demo/pycharm_eval_stateful_iterator_resume_bug_4007/README.md)
  - edge: Rainbow solved 4/4 with fewer tokens/time; random only reached 3/4 tests
  - family: resumable iterator state restore
- [pycharm_synthetic_hard_cache_invalidation_9001](/Users/aditya/Developer/hyperreasoning/demo/pycharm_synthetic_hard_cache_invalidation_9001/README.md)
  - edge: synthetic hard candidate, not yet proven
  - family: dependency invalidation cache

## Rainbow Solved, All Competitors Failed

No tasks in the current same-pipeline demo-folder scan met this stricter bar.
Older scans had such cases, but rerunning against the exact demo folders showed
competitors solving or partially solving those tasks.

Do not claim this category in the live demo unless a new same-pipeline scan
reproduces it.

## Rainbow Solved, At Least One Competitor Failed

This is the strongest current solve-quality edge.

| Demo | Rainbow | Failed competitor signal |
| --- | --- | --- |
| [ast_transform_scope_bug_2058](/Users/aditya/Developer/hyperreasoning/demo/pycharm_eval_ast_transform_scope_bug_2058/README.md) | 4/4 solved, 736 tokens, 5.7s | one-shot failed at 3/4, 4613 tokens, 74.2s |

## Same Quality, Rainbow More Efficient

These show the efficiency story when all selected competitors also solve.

| Demo | Rainbow | Competitor signal |
| --- | --- | --- |
| [cache_invalidation_dependency_2073](/Users/aditya/Developer/hyperreasoning/demo/pycharm_eval_cache_invalidation_dependency_2073/README.md) | 3/3 solved, 745 tokens, 5.6s | all competitors solved, but used more tokens/time |
| [multi_file_interface_drift_4025](/Users/aditya/Developer/hyperreasoning/demo/pycharm_eval_multi_file_interface_drift_4025/README.md) | 5/5 solved, 1159 tokens, 6.5s | all competitors solved, but used more tokens/time |

## Secondary Partial-Quality Edges

These are useful backups, but not the headline proof claim.

| Demo | Rainbow | Competitor signal |
| --- | --- | --- |
| [stateful_iterator_resume_bug_1080](/Users/aditya/Developer/hyperreasoning/demo/pycharm_eval_stateful_iterator_resume_bug_1080/README.md) | 4/4 solved, 1311 tokens, 25.4s | heuristic reached only 3/4 |
| [stateful_iterator_resume_bug_4007](/Users/aditya/Developer/hyperreasoning/demo/pycharm_eval_stateful_iterator_resume_bug_4007/README.md) | 4/4 solved, 1310 tokens, 24.0s | random reached only 3/4 |

Notes:

- This folder intentionally keeps the five highest-signal demos from the
  same-pipeline demo-folder scan.
- The proof metrics above come from `.hyper/rainbow_edge_scans/demo_folder_records_latest.jsonl`.
- Every demo includes the original `task.json`, visible/hidden tests, and a
  `reference/` solution snapshot.
- Synthetic demos are included as stress-test candidates only. Do not claim
  them as Rainbow wins until a same-pipeline scan confirms the edge.
