import tensorflow as tf
import numpy as np
from MNIST import *
from homomorphisms import *

def relative_error(actual, measured):
	return np.mean(np.abs((actual-measured)/actual))

def _randomize_parameters(params):
	for i in params:
		for j in params[i]:
			s = params[i][j].shape
			params[i][j] += np.random.random(s) - 0.5
			params[i][j] *= np.random.random(s) - 0.5
	return params

def get_random_model():
	''' Make random MNIST model '''
	model = MNIST_model(random_slope=True)
	model.ensure_session()
	random_params = _randomize_parameters(model.get_layers())
	model.set_layers(random_params)
	return model

def test_numpy_get_set():
	model = get_random_model()
	X = tf.expand_dims(tf.reshape(tf.range(np.prod(model.input_shape), \
		dtype=tf.float32), model.input_shape), axis=0)
	before = model.sess.run(model.forward(X))
	params = model.get_layers()
	model.set_layers(params)
	after = model.sess.run(model.forward(X))
	diff = relative_error(before, after)
	if diff<1e-8:
		print("Passed TensorFlow <-> Numpy getting/setting")
		return True
	else:
		print("Failed TensorFlow <-> Numpy getting/setting")
		return False

def test_random_to_identity_relu():
	model = get_random_model()
	X = tf.expand_dims(tf.reshape(tf.range(np.prod(model.input_shape), \
		dtype=tf.float32), model.input_shape), axis=0)
	before = model.sess.run(model.forward(X))
	params = model.get_layers()
	params_to_identity_relu(params)
	model.set_layers(params)
	after = model.sess.run(model.forward(X))
	diff = relative_error(before, after)
	if diff<1e-5:
		print("Passed Random -> Identity ReLu conversion")
		return True
	else:
		print("Failed Random -> Identity ReLu conversion")
		return False

def test_neuron_permutation():
	model = get_random_model()
	X = tf.expand_dims(tf.reshape(tf.range(np.prod(model.input_shape), \
		dtype=tf.float32), model.input_shape), axis=0)
	before = model.sess.run(model.forward(X))
	params = model.get_layers()
	perm = get_identity_permutation(params)
	for i in range(len(perm)): np.random.shuffle(perm[i])
	apply_permutation_to_params(params, perm)
	model.set_layers(params)
	after = model.sess.run(model.forward(X))
	diff = relative_error(before, after)
	#print("Difference: ", diff, "\n", before, "\n", after)
	if diff<1e-5:
		print("Passed Neuron Shuffle")
		return True
	else:
		print("Failed Neuron Shuffle")
		return False

def run_all_tests():
	test_numpy_get_set()
	test_random_to_identity_relu()
	test_neuron_permutation()

if __name__ == '__main__':
	run_all_tests()
