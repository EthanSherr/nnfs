import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from Layer_Dense import Layer_Dense
from Activation_SoftMax import Activation_SoftMax
from Activation_ReLU import Activation_ReLU
from Loss import Loss_CategoricalCrossentropy

nnfs.init()

X, y = spiral_data(samples=100, classes=3)

print('X', X[:5])
print('y', y)