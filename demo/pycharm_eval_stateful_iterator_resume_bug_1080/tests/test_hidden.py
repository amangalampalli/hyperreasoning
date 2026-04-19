import unittest

from iterator_impl import CheckpointingIterator


class IteratorHiddenTests(unittest.TestCase):
    def test_exhausted_checkpoint_stays_exhausted(self) -> None:
        iterator = CheckpointingIterator([[1]])
        self.assertEqual(list(iterator), [1])
        checkpoint = iterator.checkpoint()
        resumed = CheckpointingIterator.from_checkpoint([[1]], checkpoint)
        self.assertEqual(list(resumed), [])

    def test_constructor_copies_nested_groups(self) -> None:
        groups = [[10, 20], [30]]
        iterator = CheckpointingIterator(groups)
        checkpoint = iterator.checkpoint()
        groups[0].append(999)
        resumed = CheckpointingIterator.from_checkpoint(groups, checkpoint)
        self.assertEqual(list(resumed), [10, 20, 30])

if __name__ == "__main__":
    unittest.main()
