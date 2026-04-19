import unittest

from client import average_score, list_ids


RECORDS = [
    {"id": "a", "score": 5, "active": True},
    {"id": "b", "score": 2, "active": False},
    {"id": "c", "score": 8, "active": True},
]


class InterfaceVisibleTests(unittest.TestCase):
    def test_list_ids_filters_inactive_by_default(self) -> None:
        self.assertEqual(list_ids(RECORDS), ["a", "c"])

    def test_average_score_respects_threshold(self) -> None:
        self.assertEqual(average_score(RECORDS, min_score=6), 8.0)
