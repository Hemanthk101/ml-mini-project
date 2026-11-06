import unittest
from batching import make_batches

class BatchTests(unittest.TestCase):
    def test_frame_budget_and_every_example_preserved(self):
        examples=[('a',None,None,100),('b',None,None,200),('c',None,None,620),('d',None,None,300)]
        batches=make_batches(examples,3,5,900)
        self.assertEqual(sorted(x[0] for b in batches for x in b),['a','b','c','d'])
        self.assertTrue(all(len(b)*((max(x[-1] for x in b)+5)//5*5)<=900 for b in batches))
    def test_oversize_sample_is_explicit_error(self):
        with self.assertRaises(ValueError): make_batches([('a',None,None,1000)],2,5,900)
    def test_pad_only_to_local_batch_maximum(self):
        batches=make_batches([('a',None,None,10),('b',None,None,12),('c',None,None,1000)],2,5)
        self.assertEqual([[x[-1] for x in b] for b in batches],[[10,12],[1000]])
