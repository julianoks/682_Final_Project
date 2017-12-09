import tensorflow as tf
from numpy.random import RandomState
import numpy as np

class Classifier_Model(object):
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
		scores = tf.map_fn(lambda x: self.forward([x])[0], X)
		correct_prediction = tf.equal(tf.argmax(scores, 1), tf.argmax(Y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		return accuracy

	def ensure_session(self):
		if (self.sess == None) or self.sess._closed:
			self.sess = tf.Session()
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
		variable_objects = [v for v in tf.trainable_variables()]
		gradient_objects = [tf.gradients(loss, [v])[0] for v in variable_objects]
		zipped = list(zip(variable_objects, gradient_objects))
		zipped = list(filter(lambda x:x[1]!=None, zipped))
		# start training
		self.ensure_session()
		if init_params != None:
			for k in self.layers:
				self.layers[k].slope = tf.cast(init_params[k]['slope'], dtype=tf.float32)
		rand_state = RandomState(random_seed)
		history = []
		for i in range(iterations):
			batch = mnist_dataset.train.next_batch(batch_size)
			var, grad = zipped[int(rand_state.rand() * len(zipped))]
			var_mask = tf.cast(rand_state.rand(*var.shape)<update_percent, var.dtype)
			#var_mask = np.zeros(shape=var.shape)
			#for _ in range(3): var_mask[:,:,:,int(rand_state.rand()*int(var.shape[-1]))] = 1
			#var_mask = tf.cast(var_mask, var.dtype)
			var_grad = self.sess.run(grad, feed_dict={X: batch[0], Y: batch[1]})
			if (print_every != False) and (i % print_every == 0):
				batch_accuracy = self.sess.run(accuracy, feed_dict={X: batch[0], Y: batch[1]})
				history.append(batch_accuracy)
				print('step %d, batch accuracy %g' % (i, batch_accuracy))
			new_val = var - (learning_rate * var_mask * var_grad)
			assn_preval = tf.assign(var, new_val)
			self.sess.run(assn_preval)
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
