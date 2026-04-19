from __future__ import annotations

from api import summarize_records


def list_ids(
    records: list[dict[str, object]],
    *,
    include_inactive: bool = False,
    min_score: int = 0,
) -> list[str]:
    summary = summarize_records(
        records,
        include_inactive=include_inactive,
        min_score=min_score,
    )
    return list(summary["ids"])


def average_score(
    records: list[dict[str, object]],
    *,
    include_inactive: bool = False,
    min_score: int = 0,
) -> float:
    summary = summarize_records(
        records,
        include_inactive=include_inactive,
        min_score=min_score,
    )
    if summary["count"] == 0:
        return 0.0
    return summary["total_score"] / summary["count"]
