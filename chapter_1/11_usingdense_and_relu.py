import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from Layer_Dense import Layer_Dense, Activation_ReLU
# import matplotlib.pyplot as plt

nnfs.init()

X, y = spiral_data(samples=100, classes=3)
print('X\n', X)


# 2 input features and 3 output values
dense1 = Layer_Dense(2, 3)
dense1.forward(X)
print('dense1 output', dense1.output[:5])

activation1 = Activation_ReLU(dense1.output)
print('dense1 relu', activation1.output[:5])