import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib

devices = device_lib.list_local_devices()
gpus = [d for d in devices if d.device_type == 'GPU']
if not gpus:
    raise SystemExit('FAIL: no TensorFlow CUDA GPU available.')
with tf.device('/gpu:0'):
    a = tf.constant([[1.,2.],[3.,4.]])
    product = tf.matmul(a, a)
    image = tf.ones([1,8,8,1])
    kernel = tf.ones([3,3,1,1])
    convolution = tf.nn.conv2d(image, kernel, [1,1,1,1], 'VALID')
    gradient = tf.gradients(tf.reduce_sum(convolution), kernel)[0]
config = tf.ConfigProto(allow_soft_placement=False)
config.gpu_options.allow_growth = True
with tf.Session(config=config) as session:
    matrix, conv, grad = session.run([product, convolution, gradient])
assert np.allclose(matrix, [[7,10],[15,22]])
assert np.allclose(conv,9) and np.allclose(grad,36)
report = dict(tensorflow=tf.__version__, cuda_build=tf.test.is_built_with_cuda(),
    devices=[dict(name=d.name, description=d.physical_device_desc) for d in gpus],
    matrix_device=product.device, convolution_device=convolution.device,
    strict_gpu_placement=True, matmul_passed=True, convolution_and_gradient_passed=True)
Path('results').mkdir(exist_ok=True)
Path('results/gpu_check.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
