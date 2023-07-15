import numpy as np

inputs1 = [1.0, 2.0, 3.0, 2.5]
weights2 = [0.2, 0.8, -0.5, 1.0]
bias = 2.0

print('np.dot(list of inputs, list of weights) + bias num', np.dot(weights2, inputs1) + bias)



inputs = [1.0, 2.0, 3.0, 2.5]
weights = [[0.2, 0.8, -0.5, 1], [0.5, -0.91, 0.26, -0.5], [-0.26, -0.27, 0.17, 0.87]]
biases = [2.0, 3.0, 0.5]

# whatever coems first decides the output shape - thus matrix goes first
layer_outputs = np.dot(weights, inputs) + biases
print(layer_outputs)

manual = [np.dot(weights[0], inputs) + biases[0], np.dot(weights[1], inputs) + biases[1], np.dot(weights[2], inputs)  + biases[2]]
print('a manual dot product ...[np.dot(weights[0], inputs) + biases[0], np.dot(weights[1], inputs) + biases[1], np.dot(weights[2], inputs)  + biases[2]]', manual)

print('np.array([1,2,3])', np.array([1,2,3]))

print('np.dot(np.array([[1,2,3]]), [3,4,5])', np.dot(np.array([[1,2,3]]), [3,4,5]))

# np.expand_dims() adds a new dimension at the index of the axis.
print("np.expand_dims(np.array([1,2,3], axis=0)) = ", np.expand_dims(np.array([1,2,3]), axis=0))

# shape(n,1)
print('np.array([[1,2,3]]).transpose()', np.array([[1,2,3]]).transpose())
print('np.array([[1,2,3]]).T', np.array([[1,2,3]]).T)
