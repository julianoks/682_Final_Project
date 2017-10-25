import numpy as np

def params_to_identity_relu(params):
	for k in params:
		params[k]['W'] *= params[k]['slope']
		params[k]['b'] *= params[k]['slope']
		params[k]['slope'] = np.ones_like(params[k]['slope'])
	return params

def get_identity_permutation(params):
	layers = sorted(params.keys())[:-1]
	permutation = []
	for k in layers:
		permutation.append(np.arange(params[k]['b'].shape[-1]))
	return permutation

def apply_permutation_to_params(params, permutation):
	layers = sorted(params.keys())
	for i, perm in enumerate(permutation):
		a, b = params[layers[i]], params[layers[i+1]]
		a['W'] = a['W'][:,:,:,perm]
		a['b'] = a['b'][:,:,:,perm]
		a['slope'] = a['slope'][:,:,:,perm]
		b['W'] = b['W'][:,:,perm,:]
	return params
