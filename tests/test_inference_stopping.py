"""Regression tests for EOS patience and reusable TensorFlow attention history."""
import unittest
from types import SimpleNamespace
import numpy as np
import tensorflow as tf
from models.helpers import TacoTestHelper

class StopTests(unittest.TestCase):
    def tearDown(self):
        tf.reset_default_graph()
    def build(self, weights, times, controlled=True):
        history=None
        if controlled:
            rows=tf.constant(np.asarray(weights,dtype=np.float32)[:,None,:])
            history=tf.TensorArray(tf.float32,size=len(weights),clear_after_read=False).unstack(rows)
        state=(SimpleNamespace(alignment_history=history),)
        helper=TacoTestHelper(tf.constant(1),2,1,tf.constant([len(weights[0])]) if controlled else None)
        return [helper.next_inputs(tf.constant(t),tf.ones([1,2]),state,None)[0] for t in times]
    def evaluate(self, tensors):
        with tf.Session(config=tf.ConfigProto(device_count={'GPU':0},intra_op_parallelism_threads=1,inter_op_parallelism_threads=1)) as session:
            return [bool(x[0]) for x in session.run(tensors)]
    def test_single_jump_does_not_stop(self):
        a=np.tile([1,0,0,0,0],(22,1));a[-1]=[0,0,0,0,1]
        self.assertEqual(self.evaluate(self.build(a,[21])),[False])
    def test_sustained_eos_and_overlapping_reads(self):
        a=np.tile([1,0,0,0,0],(22,1));a[13:]=[0,0,0,0,1]
        self.assertEqual(self.evaluate(self.build(a,[20,21])),[True,True])
    def test_minimum_length_guard(self):
        a=np.tile([0,0,0,0,1],(10,1))
        self.assertEqual(self.evaluate(self.build(a,[9])),[False])
    def test_default_policy_is_unchanged(self):
        a=np.tile([0,0,0,0,1],(22,1))
        self.assertEqual(self.evaluate(self.build(a,[21],False)),[False])
