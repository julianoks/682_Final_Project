import tensorflow as tf
from utils import *

class MNIST_model(object):
	def __init__(self, random_slope=False, reg_strength=1e-4):
		self.sess = None
		self.input_shape = [784]
		self.random_slope = random_slope
		self.reg_strength = reg_strength
		self.layers = {}
		# input (?, 784)
		# reshape (?, 28, 28, 1)
		# layer 1 (?, 28, 28, 32)
		self.layers[1] = conv_layer(filter_size=[5,5], input_channels=1, n_filters=32, random_slope=random_slope)
		# layer 2 (?, 28, 28, 32)
		self.layers[2] = conv_layer(filter_size=[2,2], input_channels=32, n_filters=32, random_slope=random_slope)
		# max pool (?, 14, 14, 32)
		# layer 3 (?, 14, 14, 64)
		self.layers[3] = conv_layer(filter_size=[2,2], input_channels=32, n_filters=64, random_slope=random_slope)
		# max pool (?, 7, 7, 64)
		# layer 4 (?, 1, 1, 1024)
		self.layers[4] = conv_layer(filter_size=[7,7], input_channels=64, n_filters=1024, random_slope=random_slope)
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

	def loss(self, X, Y):
		scores = self.forward(X)
		cross_entropy = tf.reduce_mean( \
			tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=scores))
		reg_loss = 0
		for k in self.layers:
			reg_loss += self.reg_strength * tf.reduce_sum(self.layers[k].W ** 2)
		loss = cross_entropy + reg_loss
		return loss

	def accuracy(self, X, Y):
		scores = self.forward(X)
		correct_prediction = tf.equal(tf.argmax(scores, 1), tf.argmax(Y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		return accuracy

	def make_session(self):
		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer())

	def close_session(self):
		if self.sess != None: self.sess.close()

	def train(self, mnist_dataset, iterations=1000, batch_size=50, print_every=100):
		X = tf.placeholder(tf.float32, shape=[None, 784])
		Y = tf.placeholder(tf.float32, shape=[None, 10])
		loss = self.loss(X, Y)
		train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
		accuracy = self.accuracy(X, Y)
		# start training
		if (self.sess == None) or self.sess._closed:
			self.make_session()
		for i in range(iterations):
			batch = mnist_dataset.train.next_batch(batch_size)
			if (print_every != False) and (i % print_every == 0):
				train_accuracy = self.sess.run(accuracy, feed_dict={X: batch[0], Y: batch[1]})
				print('step %d, training accuracy %g' % (i, train_accuracy))
			train_step.run(feed_dict={X: batch[0], Y: batch[1]}, session=self.sess)
		print('test accuracy %g' % self.sess.run(accuracy, feed_dict= \
			{X: mnist_dataset.test.images, Y: mnist_dataset.test.labels}))

if __name__ == '__main__':
	mnist_dataset = get_mnist_dataset()
	model = MNIST_model()
	model.train(mnist_dataset)

