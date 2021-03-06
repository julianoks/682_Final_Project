import tensorflow as tf
from numpy.random import RandomState
import numpy as np

class Classifier_Model(object):
	def loss(self, X, Y):
		scores = self.forward(X)
		cross_entropy = tf.reduce_mean( \
			tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=scores))
		reg_loss = 0
		if hasattr(self, 'reg_penalties'):
			for k in self.layers:
				reg_loss += tf.reduce_sum(self.reg_penalties[k] * (self.layers[k].W ** 2))
		else:
			for k in self.layers:
				reg_loss += tf.abs(self.reg_strength * tf.reduce_sum(self.layers[k].W ** 2))
		loss = cross_entropy + reg_loss
		return loss

	def accuracy(self, X, Y):
		scores = tf.map_fn(lambda x: self.forward([x])[0], X)
		correct_prediction = tf.equal(tf.argmax(scores, 1), tf.argmax(Y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		return accuracy

	def ensure_session(self):
		if (self.sess == None) or self.sess._closed:
			config = tf.ConfigProto()
			config.gpu_options.allow_growth=True
			self.sess = tf.Session(config=config)
			self.sess.run(tf.global_variables_initializer())

	def close_session(self):
		if self.sess != None: self.sess.close()

	def train(self, mnist_dataset, iterations=1000, init_params=None, batch_size=64, print_every=100, learning_rate=1e-3):
		''' init_params can be None, or hold the values for the inititial params, but only uses them them to initialize the slopes '''
		X = tf.placeholder(tf.float32, shape=[None, 784])
		Y = tf.placeholder(tf.float32, shape=[None, 10])
		loss = self.loss(X, Y)
		train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
		accuracy = self.accuracy(X, Y)
		# start training
		self.ensure_session()
		if init_params != None:
			for k in self.layers:
				self.layers[k].slope = tf.cast(init_params[k]['slope'], dtype=tf.float32)
		history = []
		for i in range(iterations):
			batch = mnist_dataset.train.next_batch(batch_size)
			if (print_every != False) and (i % print_every == 0):
				train_accuracy = self.sess.run(accuracy, feed_dict={X: batch[0], Y: batch[1]})
				print('step %d, training accuracy %g' % (i, train_accuracy))
				history.append(train_accuracy)
			train_step.run(feed_dict={X: batch[0], Y: batch[1]}, session=self.sess)
		if print_every != False:
			print('test accuracy %g' % self.sess.run(accuracy, feed_dict= \
				{X: mnist_dataset.test.images, Y: mnist_dataset.test.labels}))
			print(history)
		return history


	def train_sheduled_sparse(self, mnist_dataset, iterations=1000, init_params=None, batch_size=64, print_every=100, learning_rate=1e-3, random_seed=123456789, update_percent=0.1):
		''' init_params can be None, or hold the values for the inititial params, but only uses them them to initialize the slopes '''
		X = tf.placeholder(tf.float32, shape=[None, 784])
		Y = tf.placeholder(tf.float32, shape=[None, 10])
		loss = self.loss(X, Y)
		accuracy = self.accuracy(X, Y)
		# start training
		self.ensure_session()
		if init_params != None:
			for k in self.layers:
				self.layers[k].slope = tf.cast(init_params[k]['slope'], dtype=tf.float32)
		rand_state = RandomState(random_seed)
		history = []
		for i in range(iterations):
			batch = mnist_dataset.train.next_batch(batch_size)
			layer = self.layers[sorted(self.layers.keys())[int(rand_state.rand() * len(self.layers.keys()))]]
			inds = np.arange(int(layer.b.shape[-1]))
			inds = rand_state.choice(inds, int(np.ceil(update_percent*len(inds))), replace=False)
			mask = np.zeros(layer.b.shape)
			mask[:,:,:,inds] = 1
			grad = tf.gradients(loss, [layer.W, layer.b])
			wg,bg = self.sess.run(grad, feed_dict={X: batch[0], Y: batch[1]})
			assns = []
			assns.append(tf.assign(layer.W, layer.W - (learning_rate * wg * mask)))
			assns.append(tf.assign(layer.b, layer.b - (learning_rate * bg * mask)))
			self.sess.run(assns)
			if (print_every != False) and (i % print_every == 0):
				batch_accuracy = self.sess.run(accuracy, feed_dict={X: batch[0], Y: batch[1]})
				history.append(batch_accuracy)
				print('step %d, batch accuracy %g' % (i, batch_accuracy))
		if (print_every != False):
			print('test accuracy %g' % self.sess.run(accuracy, feed_dict= \
				{X: mnist_dataset.test.images, Y: mnist_dataset.test.labels}))
			print(history)
		return history

	def get_layers(self):
		numpys = {}
		for k in self.layers:
			numpys[k] = {'W': self.sess.run(self.layers[k].W),
						'b': self.sess.run(self.layers[k].b),
						'slope': self.sess.run(self.layers[k].slope)}
		return numpys

	def set_layers(self, numpys):
		assns = []
		for k in numpys:
			assns.append(self.layers[k].W.assign(numpys[k]['W']))
			assns.append(self.layers[k].b.assign(numpys[k]['b']))
			self.layers[k].slope = tf.cast(numpys[k]['slope'], dtype=tf.float32)
		self.sess.run(assns)
