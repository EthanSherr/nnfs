import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from Layer_Dense import Layer_Dense
from Activation_SoftMax import Activation_SoftMax
from Activation_ReLU import Activation_ReLU
from Loss import Loss_CategoricalCrossentropy

nnfs.init()

X, y = spiral_data(samples=100, classes=3)

dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(3, 3)
activation2 = Activation_SoftMax()

loss_function = Loss_CategoricalCrossentropy()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense1.output)

loss = loss_function.calculate(activation2.output, y)

print(activation2.output[:5])
print('loss:', loss)


# updated with accuracy which averages hits & misses across samples
predictions = np.argmax(activation2.output, axis=1)
if len(y.shape) == 2: # never the case, why bother;  good example though for non sparse target_values
    y = np.argmax(y, axis=1)
accuracy = np.mean(predictions == y)

print('accuracy: ', accuracy)