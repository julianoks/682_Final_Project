import tensorflow as tf
import numpy as np

def train_pair(dataset, model_class, n_iterations=1000, random_slope=False, reg_strength=1e-4, printing=True):
	dummy = model_class(random_slope=random_slope)
	dummy.ensure_session()
	init_params = dummy.get_layers()
	dummy.close_session()
	del dummy

	if printing: print("Training model 1...")
	model1 = model_class(random_slope=random_slope)
	model1.train(dataset, iterations=n_iterations, init_params=init_params)
	if printing: print("Training model 2...")
	model2 = model_class(random_slope=random_slope)
	model2.train(dataset, iterations=n_iterations, init_params=init_params)
	return model1, model2



def serialize_weights(model):
	params = model.get_layers()
	serial = []
	for i in sorted(params.keys()):
		for j in sorted(params[i].keys()):
			serial.append(params[i][j].ravel())
	return np.hstack(serial)

def relative_difference_between_nets(model1, model2):
	s1, s2 = serialize_weights(model1), serialize_weights(model2)
	return np.sum(np.abs(((s1+s2)*(s1-s2)) / (s1*s2)))


def get_accuracy(model, X, Y):
	fed_X = tf.placeholder(tf.float32, shape=X.shape)
	fed_Y = tf.placeholder(tf.float32, shape=Y.shape)
	accuracy = model.accuracy(fed_X, fed_Y)
	return model.sess.run(accuracy, feed_dict={fed_X: X, fed_Y: Y})

def recombine(model_class, model1, model2):
	params1 = model1.get_layers()
	params2 = model2.get_layers()
	combined_params = {}
	for k in params1:
		if np.random.random() < 0.5:
			combined_params[k] = params1[k]
		else: combined_params[k] = params2[k]
	combined_model = model_class(random_slope=model1.random_slope, reg_strength=model1.reg_strength)
	combined_model.ensure_session()
	combined_model.set_layers(combined_params)
	return combined_model

def recomb_accuracy(dataset, model_class, n_recombinations=10, n_iterations=1000, random_slope=False, reg_strength=1e-4, printing=True):
	if printing: print("Using random slope?", random_slope)
	model1, model2 = train_pair(dataset, model_class, n_iterations=n_iterations, random_slope=random_slope, reg_strength=reg_strength, printing=printing)
	print("relative_difference_between_nets:", relative_difference_between_nets(model1, model2))
	child = recombine(model_class, model1, model2)
	X, Y = dataset.test.images, dataset.test.labels
	def recombined_accuracy():
		child = recombine(model_class, model1, model2)
		model1.close_session()
		model2.close_session()
		return get_accuracy(child, X, Y)
	accuracies = [recombined_accuracy() for _ in range(n_recombinations)]
	print("Recombined Accuracies:", accuracies , "Mean:", np.mean(accuracies), "\n\n")
	return np.mean(accuracies), accuracies



if __name__ == "__main__":
	from utils import get_mnist_dataset
	from MNIST import MNIST_model
	mnist = get_mnist_dataset()
	recomb_accuracy(mnist, MNIST_model, random_slope=True, n_iterations=500, n_recombinations=5)
	recomb_accuracy(mnist, MNIST_model, random_slope=False, n_iterations=500, n_recombinations=5)
