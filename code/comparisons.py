import tensorflow as tf
import numpy as np
import json
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

def inject_reg_penalties(model1, model2, strength, std):
	model1.reg_penalties, model2.reg_penalties = {}, {}
	for k in model1.layers:
		pen = strength + (std * np.random.randn(*model1.layers[k].b.shape))
		pen_tf = tf.convert_to_tensor(pen, dtype=model1.layers[k].W.dtype)
		model1.reg_penalties[k], model2.reg_penalties[k] = pen_tf, pen_tf

def train_pair(dataset, model_class, rand_pen=None, n_iterations=1000, random_slope=False, reg_strength=1e-4, print_every=None, learning_rate=0.001):
	dummy = model_class(random_slope=random_slope)
	dummy.ensure_session()
	init_params = dummy.get_layers()
	dummy.close_session()
	del dummy

	model1 = model_class(random_slope=random_slope)
	model2 = model_class(random_slope=random_slope)
	if rand_pen:
		strength, std = rand_pen
		inject_reg_penalties(model1, model2, strength, std)
	model1.reg_strength, model2.reg_strength = reg_strength, reg_strength
	print("Training model 1...")
	model1.train(dataset, iterations=n_iterations, init_params=init_params, print_every=print_every, learning_rate=learning_rate)
	print("\n\n\n\n\n\n\n\n\n\nTraining model 2...")
	model2.train(dataset, iterations=n_iterations, init_params=init_params, print_every=print_every, learning_rate=learning_rate)
	return model1, model2


def train_pair_on_schedule(dataset, model_class, rand_pen=None, n_iterations=1000, random_slope=False, reg_strength=1e-4, print_every=None, update_percent=0.1, learning_rate=5e-3):
	dummy = model_class(random_slope=random_slope)
	dummy.ensure_session()
	init_params = dummy.get_layers()
	dummy.close_session()
	del dummy

	random_seed = int(1e8*np.random.random())
	model1 = model_class(random_slope=random_slope)
	model2 = model_class(random_slope=random_slope)
	if rand_pen:
		streng, std = rand_pen
		inject_reg_penalties(model1, model2, strength, std)
	model1.reg_strength, model2.reg_strength = reg_strength, reg_strength
	print("Training model 1...")
	model1.train_sheduled_sparse(dataset, iterations=n_iterations, init_params=init_params, print_every=print_every, random_seed=random_seed, update_percent=update_percent, learning_rate=learning_rate)
	print("\n\n\n\n\n\n\n\n\n\nTraining model 2...")
	model2.train_sheduled_sparse(dataset, iterations=n_iterations, init_params=init_params, print_every=print_every, random_seed=random_seed, update_percent=update_percent, learning_rate=learning_rate)
	return model1, model2


def serialize_weights(model):
	params = model.get_layers()
	serial = []
	for i in sorted(params.keys()):
		for j in sorted(params[i].keys()):
			serial.append(params[i][j].ravel())
	return np.hstack(serial)

def distance_between_nets(model1, model2):
	s1, s2 = serialize_weights(model1), serialize_weights(model2)
	return np.linalg.norm(s1-s2)


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
		combined_params[k] = params1[k]
		inds = np.arange(len(params1[k]))
		inds = np.random.choice(inds, int(len(inds)/2), replace=False)
		for i in inds:
			combined_params[k]['b'][:,:,:,i] = params2[k]['b'][:,:,:,i]
			combined_params[k]['W'][:,:,:,i] = params2[k]['W'][:,:,:,i]
	combined_model = model_class(random_slope=model1.random_slope, reg_strength=model1.reg_strength)
	combined_model.ensure_session()
	combined_model.set_layers(combined_params)
	return combined_model

def evaluate_pair(dataset, model_class, rand_pen=None, reg_strength=1e-4, sparse_training=False, update_percent=0.1, n_recombinations=10, n_iterations=1000, random_slope=False, learning_rate=0.1, print_every=100):
	if sparse_training:
		model1, model2 = train_pair_on_schedule(dataset, model_class, rand_pen=rand_pen, reg_strength=reg_strength, n_iterations=n_iterations, update_percent=update_percent, random_slope=random_slope, print_every=print_every, learning_rate=learning_rate)
	else:
		model1, model2 = train_pair(dataset, model_class, rand_pen=rand_pen, reg_strength=reg_strength, n_iterations=n_iterations, random_slope=random_slope, print_every=print_every, learning_rate=learning_rate)
	print("distance_between_nets:", distance_between_nets(model1, model2))
	child = recombine(model_class, model1, model2)
	X, Y = dataset.test.images, dataset.test.labels
	def recombined_accuracy():
		child = recombine(model_class, model1, model2)
		return get_accuracy(child, X, Y)
	accuracies = [recombined_accuracy() for _ in range(n_recombinations)]
	for name, model in [["model1.png", model1],["model2.png", model2]]:
		weight = model.get_layers()[1]['W']
		filters = np.rollaxis(np.squeeze(weight),2,0)
		min,max = filters.min(axis=(1,2)).reshape((filters.shape[0],1,1)), filters.max(axis=(1,2)).reshape((filters.shape[0],1,1))
		images = np.array((filters-min)*(255/(max-min)), dtype='uint8')
		fig, ax = plt.subplots()
		for i in range(images.shape[0]):
			_ = fig.add_subplot(1,images.shape[0]+1,i+1)
			_ = plt.imshow(images[i], cmap='gray')
		[f.set_axis_off() for f in fig.axes]
		fig.set_size_inches([images.shape[0],1])
		plt.savefig(name)
	model1.close_session()
	model2.close_session()
	print("Recombined Accuracies:", accuracies , "Mean:", np.mean(accuracies), "\n\n")
	return np.mean(accuracies), accuracies



if __name__ == "__main__":
	from utils import get_mnist_dataset
	from MNIST import MNIST_model
	mnist = get_mnist_dataset()
	hyperparams = json.load(open('hyperparameters.json'))
	evaluate_pair(mnist, MNIST_model, rand_pen=hyperparams['rand_pen'], learning_rate=hyperparams['learning_rate'], sparse_training=hyperparams['sparse_update'], update_percent=hyperparams['update_percent'], random_slope=hyperparams['random_slope'], n_iterations=hyperparams['n_iterations'], n_recombinations=hyperparams['n_recombinations'], print_every=hyperparams['print_every'])
