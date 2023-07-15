import numpy as np
import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import vertical_data
from Layer_Dense import Layer_Dense
from Activation_ReLU import Activation_ReLU
from Activation_SoftMax import Activation_SoftMax
from Loss import Loss_CategoricalCrossentropy


nnfs.init()

# I think x is basically x,y datapoints trying to classify them in 3 ways here
X, y = vertical_data(samples=100, classes=3)
# plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap='brg')
# plt.show()

print("X", X)
print("y", y)

dense1 = Layer_Dense(2, 3) # first dense layer two inputs
activation1 = Activation_ReLU()
dense2 = Layer_Dense(3, 3) # second dense layer 3 inputs 3 outputs
activation2 = Activation_SoftMax()

loss_function = Loss_CategoricalCrossentropy()

lowest_loss = 9999999
best_dense1_weights = dense1.weights.copy()
best_dense1_biases = dense1.biases.copy()
best_dense2_weights = dense2.weights.copy()
best_dense2_biases = dense2.biases.copy()

for iteration in range(10000):
    # generate a new set of weights for iteration
    dense1.weights = 0.05 * np.random.randn(2, 3)
    dense1.biases = 0.05 * np.random.randn(1, 3)
    dense2.weights = 0.05 * np.random.randn(3, 3)
    dense2.biases = 0.05 * np.random.rand(1, 3)

    # perform forward pass of training data thru this layer
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)


    loss = loss_function.calculate(activation2.output, y)

    predictions = np.argmax(activation2.output, axis=1)
    accuracy = np.mean(predictions == y)

    if loss < lowest_loss:
        print("New set of weights found, iraterion:", iteration, 
              'loss:', loss, 'accuracy:', accuracy)
        best_dense1_weights = dense1.weights.copy()
        best_dense1_biases = dense1.biases.copy()
        best_dense2_weights = dense2.weights.copy()
        best_dense2_biases = dense2.biases.copy()
        lowest_loss = loss
    

