import argparse
import csv
import json
from datetime import datetime
import math
import os
import subprocess
import time
import tensorflow as tf
import traceback

from datasets.datafeeder import DataFeeder
from hparams import hparams, hparams_debug_string
from models import create_model
from text import sequence_to_text
from util import audio, infolog, plot, ValueWindow
from reproducibility import set_seed
from gpu_runtime import require_gpu, gpu_required, traced_run
log = infolog.log


def get_git_commit():
  subprocess.check_output(['git', 'diff-index', '--quiet', 'HEAD'])   # Verify client is clean
  commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()[:10]
  log('Git commit: %s' % commit)
  return commit


def add_stats(model):
  with tf.variable_scope('stats') as scope:
    tf.summary.histogram('linear_outputs', model.linear_outputs)
    tf.summary.histogram('linear_targets', model.linear_targets)
    tf.summary.histogram('mel_outputs', model.mel_outputs)
    tf.summary.histogram('mel_targets', model.mel_targets)
    tf.summary.scalar('loss_mel', model.mel_loss)
    tf.summary.scalar('loss_linear', model.linear_loss)
    tf.summary.scalar('learning_rate', model.learning_rate)
    tf.summary.scalar('loss', model.loss)
    gradient_norms = [tf.norm(grad) for grad in model.gradients]
    tf.summary.histogram('gradient_norm', gradient_norms)
    tf.summary.scalar('max_gradient_norm', tf.reduce_max(gradient_norms))
    return tf.summary.merge_all()


def time_string():
  return datetime.now().strftime('%Y-%m-%d %H:%M')


def train(log_dir, args):
  commit = get_git_commit() if args.git else 'None'
  checkpoint_path = os.path.join(log_dir, 'model.ckpt')
  os.makedirs(args.results_dir, exist_ok=True)
  started = time.time()
  records = []
  results_path = os.path.join(args.results_dir, 'experiment_results.csv')
  with open(results_path, 'w', newline='') as f:
    csv.writer(f).writerow(['step', 'loss', 'learning_rate'])
  input_path = os.path.join(args.base_dir, args.input)
  log('Checkpoint path: %s' % checkpoint_path)
  log('Loading training data from: %s' % input_path)
  log('Using model: %s' % args.model)
  log(hparams_debug_string())

  # Set up DataFeeder:
  coord = tf.train.Coordinator()
  with tf.variable_scope('datafeeder') as scope:
    feeder = DataFeeder(coord, input_path, hparams, args.feeder_group_batches,
                        args.queue_capacity, args.max_batch_frames)

  # Set up model:
  global_step = tf.Variable(0, name='global_step', trainable=False)
  with tf.variable_scope('model') as scope:
    model = create_model(args.model, hparams)
    model.initialize(feeder.inputs, feeder.input_lengths, feeder.mel_targets, feeder.linear_targets)
    model.add_loss()
    model.add_optimizer(global_step)
    stats = add_stats(model)

  # Bookkeeping:
  step = 0
  time_window = ValueWindow(100)
  loss_window = ValueWindow(100)
  saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=2)

  # Train!
  session_config = tf.ConfigProto(intra_op_parallelism_threads=2, inter_op_parallelism_threads=2)
  session_config.gpu_options.allow_growth = True
  with tf.Session(config=session_config) as sess:
    try:
      summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
      sess.run(tf.global_variables_initializer())

      if args.restore_step:
        # Restore from a checkpoint if the user requested it.
        restore_path = '%s-%d' % (checkpoint_path, args.restore_step)
        saver.restore(sess, restore_path)
        log('Resuming from checkpoint: %s at commit: %s' % (restore_path, commit), slack=True)
      else:
        log('Starting new training run at commit: %s' % commit, slack=True)

      feeder.start_in_session(sess)

      while not coord.should_stop():
        start_time = time.time()
        fetches = [model.loss, model.learning_rate, model.optimize]
        if gpu_required() and not records:
          (loss, learning_rate, opt), device_evidence = traced_run(sess, fetches, None,
            os.path.join(args.results_dir, 'gpu_training_trace.json'))
        else:
          loss, learning_rate, opt = sess.run(fetches)
        step = int(sess.run(global_step))
        records.append((step, float(loss), float(learning_rate)))
        with open(results_path, 'a', newline='') as f:
          csv.writer(f).writerow(records[-1])
        time_window.append(time.time() - start_time)
        loss_window.append(loss)
        message = 'Step %-7d [%.03f sec/step, loss=%.05f, avg_loss=%.05f]' % (
          step, time_window.average, loss, loss_window.average)
        log(message, slack=(step % args.checkpoint_interval == 0))
        log('Learning rate: %.9g' % learning_rate)

        if loss > 100 or math.isnan(loss):
          log('Loss exploded to %.05f at step %d!' % (loss, step), slack=True)
          raise Exception('Loss Exploded')

        if step % args.summary_interval == 0:
          log('Writing summary at step: %d' % step)
          summary_writer.add_summary(sess.run(stats), step)

        if step % args.checkpoint_interval == 0:
          log('Saving checkpoint to: %s-%d' % (checkpoint_path, step))
          saver.save(sess, checkpoint_path, global_step=step)
          log('Saving audio and alignment...')
          input_seq, spectrogram, alignment = sess.run([
            model.inputs[0], model.linear_outputs[0], model.alignments[0]])
          waveform = audio.inv_spectrogram(spectrogram.T)
          audio.save_wav(waveform, os.path.join(log_dir, 'step-%d-audio.wav' % step))
          plot.plot_alignment(alignment, os.path.join(log_dir, 'step-%d-align.png' % step),
            info='%s, %s, %s, step=%d, loss=%.5f' % (args.model, commit, time_string(), step, loss))
          log('Input: %s' % sequence_to_text(input_seq))

        if args.max_steps and len(records) >= args.max_steps:
          saved = saver.save(sess, checkpoint_path, global_step=step)
          log('Saved final smoke-test checkpoint: %s' % saved)
          summary_writer.flush()
          summary_writer.close()
          summary = dict(samples=len(feeder._metadata), steps=len(records), batch_size=hparams.batch_size,
            seed=args.seed, starting_loss=records[0][1], ending_loss=records[-1][1],
            initial_observed_learning_rate=records[0][2], execution_seconds=time.time()-started,
            checkpoint=saved, hparams=hparams.values(), feeder=dict(feeder.stats),
            purpose='pipeline smoke test, not full training')
          with open(os.path.join(args.results_dir, 'experiment_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
          feeder.stop_in_session()
          break

    except Exception as e:
      log('Exiting due to exception: %s' % e, slack=True)
      traceback.print_exc()
      coord.request_stop(e)
      raise
    finally:
      feeder.stop_in_session()


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--base_dir', default=os.path.expanduser('~/tacotron'))
  parser.add_argument('--input', default='training/train.txt')
  parser.add_argument('--model', default='tacotron')
  parser.add_argument('--name', help='Name of the run. Used for logging. Defaults to model name.')
  parser.add_argument('--hparams', default='',
    help='Hyperparameter overrides as a comma-separated list of name=value pairs')
  parser.add_argument('--restore_step', type=int, help='Global step to restore from checkpoint.')
  parser.add_argument('--summary_interval', type=int, default=100,
    help='Steps between running summary ops.')
  parser.add_argument('--checkpoint_interval', type=int, default=1000,
    help='Steps between writing checkpoints.')
  parser.add_argument('--slack_url', help='Slack webhook URL to get periodic reports.')
  parser.add_argument('--tf_log_level', type=int, default=1, help='Tensorflow C++ log level.')
  parser.add_argument('--git', action='store_true', help='If set, verify that the client is clean.')
  parser.add_argument('--seed', type=int, default=2026)
  parser.add_argument('--max_steps', type=int, default=0, help='Number of updates this run; 0 means unbounded.')
  parser.add_argument('--results_dir', default='results')
  parser.add_argument('--feeder_group_batches', type=int, default=4)
  parser.add_argument('--queue_capacity', type=int, default=2)
  parser.add_argument('--max_batch_frames', type=int, default=0,
    help='Maximum batch size times padded acoustic frames; 0 disables the budget.')
  args = parser.parse_args()
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(args.tf_log_level)
  run_name = args.name or args.model
  log_dir = os.path.join(args.base_dir, 'logs-%s' % run_name)
  os.makedirs(log_dir, exist_ok=True)
  infolog.init(os.path.join(log_dir, 'train.log'), run_name, args.slack_url)
  hparams.parse(args.hparams)
  if args.max_steps < 0:
    parser.error('--max_steps must be nonnegative')
  set_seed(args.seed)
  require_gpu()
  train(log_dir, args)


if __name__ == '__main__':
  main()
