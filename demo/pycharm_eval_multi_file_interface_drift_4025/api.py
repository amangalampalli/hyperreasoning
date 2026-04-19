from __future__ import annotations


def summarize_records(
    records: list[dict[str, object]],
    *,
    include_inactive: bool = False,
    min_score: int = 0,
) -> dict[str, object]:
    ids: list[str] = []
    total_score = 0
    for record in records:
        is_active = bool(record.get("active", True))
        if not include_inactive and not is_active:
            continue
        score = int(record["score"])
        if score < min_score:
            continue
        ids.append(str(record["id"]))
        total_score += score
    return {
        "ids": ids,
        "count": len(ids),
        "total_score": total_score,
    }
