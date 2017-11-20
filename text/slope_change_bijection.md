I'd like to make a bijection between changing the slope of each ReLU and changing the learning algorithm.

The output of a standard neuron is:
y=max((w dot x) + b, 0)
Where w and x are vector of the same length and b is a scalar.

When we make the slope=λ,
y= λ * max((w dot x) + b, 0)

Let u=((w dot x) + b), then we can calculate the gradients for w, x, b given dL/du.  Note that dL/du is scaled by λ, which we bring out for clarity:
dL/dx = λ * dL/du * du/dx = λ * dL/du * w
dL/dw = λ * dL/du * du/dw = λ * dL/du * x
dL/db = λ * dL/du

This contrasts with the gradients of the standard neurons by multiplying each gradient by λ.
If one is using standard gradient descent, where x'= x - learning_rate * dL/dx, using ReLU scaling the update becomes x'= x - λ * learning_rate * dL/dx.
You can see that this only affects the parameter update by scaling the learning rate by λ.  If one is using adaptive learning rates, like RMSProp or Adam, then the adaptive rate will 'learn' the multiplicative inverse of λ, 1/λ.  I have not proven this convergence, but I think it is true because we essentially divide the gradient by a running average of its recent magnitude, which is scalled by λ.

So that takes care of the updates to w and b, but how about the gradient of dL/dx, which will also be used in gradient calculation of preceding parameters?  It too will be scaled by λ.  Well, if we assume for the standard neuron there is one optimal w and b, lets call them w* and b*, that has been learned, then in the sloped neuron the optimal w and be will be (1/λ)w*, (1/λ)b*.  So once w gets close to (1/λ)w*, dL/dx will be close to the dL/dx calculated using the optimal standard neuron.

In conclusion, changing the slope to λ will:
- Scale the initial parameters, w and b, by λ
- Scale the learning rates for w and b by λ.  If using an adaptive learning rate method, the method will remove the scaling, essentially giving all neurons the same learning rate.
- Once w and b adapt to their scaling, it will not affect the gradient w.r.t. the neuron.

