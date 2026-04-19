import unittest

from api import summarize_records
from client import average_score, list_ids


RECORDS = [
    {"id": "a", "score": 5, "active": True},
    {"id": "b", "score": 2, "active": False},
    {"id": "c", "score": 8, "active": True},
]


class InterfaceHiddenTests(unittest.TestCase):
    def test_api_uses_documented_keyword_and_keys(self) -> None:
        summary = summarize_records(RECORDS, include_inactive=True)
        self.assertEqual(summary, {"ids": ["a", "b", "c"], "count": 3, "total_score": 15})

    def test_empty_result_uses_zero_average(self) -> None:
        self.assertEqual(average_score(RECORDS, min_score=99), 0.0)

    def test_client_and_api_stay_consistent(self) -> None:
        summary = summarize_records(RECORDS, include_inactive=False, min_score=5)
        self.assertEqual(list_ids(RECORDS, min_score=5), summary["ids"])
