import tensorflow as tf

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
		scores = self.forward(X)
		correct_prediction = tf.equal(tf.argmax(scores, 1), tf.argmax(Y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		return accuracy

	def ensure_session(self):
		if (self.sess == None) or self.sess._closed:
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
		self.ensure_session()
		for i in range(iterations):
			batch = mnist_dataset.train.next_batch(batch_size)
			if (print_every != False) and (i % print_every == 0):
				train_accuracy = self.sess.run(accuracy, feed_dict={X: batch[0], Y: batch[1]})
				print('step %d, training accuracy %g' % (i, train_accuracy))
			train_step.run(feed_dict={X: batch[0], Y: batch[1]}, session=self.sess)
		print('test accuracy %g' % self.sess.run(accuracy, feed_dict= \
			{X: mnist_dataset.test.images, Y: mnist_dataset.test.labels}))

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
