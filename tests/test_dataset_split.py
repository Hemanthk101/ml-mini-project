import unittest
from split_dataset import split_rows
class SplitTests(unittest.TestCase):
    def test_duplicates_stay_together_and_split_is_repeatable(self):
        rows=['a|m|10|hello','b|m|10|HELLO','c|m|10|world','d|m|10|another','e|m|10|last']
        result=split_rows(rows)
        self.assertEqual(result,split_rows(rows))
        self.assertEqual(sorted(x for v in result.values() for x in v),sorted(rows))
        locations=[k for k,v in result.items() if rows[0] in v or rows[1] in v]
        self.assertEqual(len(locations),1)
    def test_too_few_groups_is_error(self):
        with self.assertRaises(ValueError):split_rows(['a|m|10|hello'])
