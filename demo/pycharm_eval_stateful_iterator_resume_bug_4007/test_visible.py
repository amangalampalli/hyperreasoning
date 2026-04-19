import unittest

from iterator_impl import CheckpointingIterator


class IteratorVisibleTests(unittest.TestCase):
    def test_resume_from_mid_group(self) -> None:
        iterator = CheckpointingIterator([[1, 2, 3], [4]])
        self.assertEqual(next(iterator), 1)
        checkpoint = iterator.checkpoint()
        resumed = CheckpointingIterator.from_checkpoint([[1, 2, 3], [4]], checkpoint)
        self.assertEqual(list(resumed), [2, 3, 4])

    def test_empty_groups_are_skipped(self) -> None:
        iterator = CheckpointingIterator([[], [5], [], [6, 7]])
        self.assertEqual(list(iterator), [5, 6, 7])
