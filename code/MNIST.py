import tensorflow as tf
from utils import *
from Model_base import Classifier_Model

class MNIST_model(Classifier_Model):
	def __init__(self, random_slope=False, reg_strength=1e-4):
		filter_sizes = [4,8,16] #[32, 32, 64]
		self.sess = None
		self.input_shape = [784]
		self.random_slope = random_slope
		self.reg_strength = reg_strength
		self.layers = {}
		# input (?, 784)
		# reshape (?, 28, 28, 1)
		# layer 1 (?, 28, 28, 32)
		self.layers[1] = conv_layer(filter_size=[5,5], input_channels=1, n_filters=filter_sizes[0], random_slope=random_slope)
		# layer 2 (?, 28, 28, 32)
		self.layers[2] = conv_layer(filter_size=[2,2], input_channels=filter_sizes[0], n_filters=filter_sizes[1], random_slope=random_slope)
		# max pool (?, 14, 14, 32)
		# layer 3 (?, 14, 14, 64)
		self.layers[3] = conv_layer(filter_size=[2,2], input_channels=filter_sizes[1], n_filters=filter_sizes[2], random_slope=random_slope)
		# max pool (?, 7, 7, 64)
		# layer 4 (?, 1, 1, 1024)
		self.layers[4] = conv_layer(filter_size=[7,7], input_channels=filter_sizes[2], n_filters=1024, random_slope=random_slope)
		# layer 5, w/o ReLu (?, 1, 1, 10)
		self.layers[5] = conv_layer(filter_size=[1,1], input_channels=1024, n_filters=10, random_slope=random_slope, dont_relu=True)
		# reshape (?, 10)

	def forward(self, X):
		net = tf.reshape(X, [-1, 28, 28, 1])
		net = self.layers[1].forward(net)
		net = self.layers[2].forward(net)
		net = max_pool_2x2(net)
		net = self.layers[3].forward(net)
		net = max_pool_2x2(net)
		net = self.layers[4].forward(net)
		net = self.layers[5].forward(net)
		net = net[:,0,0,:]
		return net


if __name__ == '__main__':
	mnist_dataset = get_mnist_dataset()
	model = MNIST_model()
	model.train(mnist_dataset)

