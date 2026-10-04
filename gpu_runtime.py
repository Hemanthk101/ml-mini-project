"""Verify GPU use through actual TensorFlow run metadata."""
import json
import os
import re
from pathlib import Path
import tensorflow as tf

def gpu_required():
    return os.environ.get('TACOTRON_REQUIRE_GPU') == '1'

def require_gpu():
    if gpu_required() and not tf.test.is_gpu_available(cuda_only=True):
        raise RuntimeError('GPU required, but TensorFlow cannot use CUDA. Run setup-gpu.ps1.')

def traced_run(session, fetches, feed_dict, evidence_path):
    metadata = tf.RunMetadata()
    value = session.run(fetches, feed_dict=feed_dict,
        options=tf.RunOptions(trace_level=tf.RunOptions.SOFTWARE_TRACE), run_metadata=metadata)
    devices = [d.device for d in metadata.step_stats.dev_stats]
    gpu_nodes = [n.node_name for d in metadata.step_stats.dev_stats if 'gpu:0' in d.device.lower()
                 for n in d.node_stats]
    # A registered device alone is insufficient: require observed model compute kernels.
    compute = [n for n in gpu_nodes if re.search(
        r'/(?:Conv2D|Conv2DBackpropInput|Conv2DBackpropFilter|MatMul|BatchMatMul)(?:_\d+)?$', n)]
    if not compute:
        raise RuntimeError('No GPU convolution/matrix/recurrent compute observed; refusing a GPU success claim.')
    evidence = dict(devices=devices, gpu_node_count=len(gpu_nodes), gpu_compute_node_count=len(compute),
                    trace_type='TensorFlow operation execution on assigned GPU device (software trace)',
                    gpu_compute_examples=compute[:40], verified_gpu_compute=True)
    path = Path(evidence_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(evidence, indent=2), encoding='utf-8')
    return value, evidence
