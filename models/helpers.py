import numpy as np
import tensorflow as tf
from tensorflow.contrib.seq2seq import Helper


# Adapted from tf.contrib.seq2seq.GreedyEmbeddingHelper
class TacoTestHelper(Helper):
  def __init__(self, batch_size, output_dim, r, input_lengths=None, eos_patience=8):
    with tf.name_scope('TacoTestHelper'):
      self._batch_size = batch_size
      self._output_dim = output_dim
      self._end_token = tf.tile([0.0], [output_dim * r])
      self._input_lengths = input_lengths
      self._eos_patience = eos_patience

  @property
  def batch_size(self):
    return self._batch_size

  @property
  def sample_ids_shape(self):
    return tf.TensorShape([])

  @property
  def sample_ids_dtype(self):
    return np.int32

  def initialize(self, name=None):
    return (tf.tile([False], [self._batch_size]), _go_frames(self._batch_size, self._output_dim))

  def sample(self, time, outputs, state, name=None):
    return tf.tile([0], [self._batch_size])  # Return all 0; we ignore them

  def next_inputs(self, time, outputs, state, sample_ids, name=None):
    '''Stop on EOS. Otherwise, pass the last output as the next input and pass through state.'''
    with tf.name_scope('TacoTestHelper'):
      finished = tf.reduce_all(tf.equal(outputs, self._end_token), axis=1)
      if self._input_lengths is not None:
        # Checkpoint-compatible inference policy: require sustained attention on
        # the actual EOS character, not a single transient jump. No new weights.
        history = state[0].alignment_history.gather(
          tf.range(tf.maximum(0, time-self._eos_patience+1), time+1))
        eos_mask = tf.one_hot(self._input_lengths-1, tf.shape(history)[-1])
        eos_mass = tf.reduce_sum(history * eos_mask[None, :, :], axis=2)
        sustained = tf.reduce_all(eos_mass >= .9, axis=0)
        minimum = tf.maximum(20, self._input_lengths // 2)
        finished = tf.logical_or(finished, tf.logical_and(time+1 >= minimum, sustained))
      # Feed last output frame as next input. outputs is [N, output_dim * r]
      next_inputs = outputs[:, -self._output_dim:]
      return (finished, next_inputs, state)


class TacoTrainingHelper(Helper):
  def __init__(self, inputs, targets, output_dim, r):
    # inputs is [N, T_in], targets is [N, T_out, D]
    with tf.name_scope('TacoTrainingHelper'):
      self._batch_size = tf.shape(inputs)[0]
      self._output_dim = output_dim

      # Feed every r-th target frame as input
      self._targets = targets[:, r-1::r, :]

      # Use full length for every target because we don't want to mask the padding frames
      num_steps = tf.shape(self._targets)[1]
      self._lengths = tf.tile([num_steps], [self._batch_size])

  @property
  def batch_size(self):
    return self._batch_size

  @property
  def sample_ids_shape(self):
    return tf.TensorShape([])

  @property
  def sample_ids_dtype(self):
    return np.int32

  def initialize(self, name=None):
    return (tf.tile([False], [self._batch_size]), _go_frames(self._batch_size, self._output_dim))

  def sample(self, time, outputs, state, name=None):
    return tf.tile([0], [self._batch_size])  # Return all 0; we ignore them

  def next_inputs(self, time, outputs, state, sample_ids, name=None):
    with tf.name_scope(name or 'TacoTrainingHelper'):
      finished = (time + 1 >= self._lengths)
      next_inputs = self._targets[:, time, :]
      return (finished, next_inputs, state)


def _go_frames(batch_size, output_dim):
  '''Returns all-zero <GO> frames for a given batch size and output dimension'''
  return tf.tile([[0.0]], [batch_size, output_dim])

