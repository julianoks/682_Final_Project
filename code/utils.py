import tensorflow as tf

def _weight_var(shape):
	var = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(var)

def _bias_var(shape, constant=0):
	var = tf.constant(float(constant), shape=shape)
	return tf.Variable(var)

def _slope_const(n_filters, random=False):
	if random:
		return tf.constant(tf.truncated_normal(shape, stddev=1))
	else:
		return tf.constant(1., shape=[1,1,1,n_filters])

class conv_layer(object):
	def __init__(self, filter_size=[5,5], input_channels=1, n_filters=32, random_slope=False, dont_relu=False):
		self.dont_relu = dont_relu
		self.W = _weight_var([*filter_size,input_channels,n_filters])
		self.b = _bias_var([n_filters])
		self.slope = _slope_const(n_filters, random=random_slope)
	def forward(self, x):
		out = tf.nn.conv2d(x, self.W, strides=[1, 1, 1, 1], padding='SAME')
		out += self.b
		out *= self.slope
		if self.dont_relu: return out
		out = tf.nn.relu(out)
		return out

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], \
  						strides=[1, 2, 2, 1], padding='SAME')

def get_mnist_dataset():
	from tensorflow.examples.tutorials.mnist import input_data
	mnist_dataset = input_data.read_data_sets('MNIST_data', one_hot=True)
	return mnist_dataset
