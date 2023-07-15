import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from Layer_Dense import Layer_Dense
# import matplotlib.pyplot as plt

nnfs.init()

X, y = spiral_data(samples=100, classes=3)
print('X\n', X)
# 2 input features and 3 output values
dense1 = Layer_Dense(2, 3)
dense1.forward(X)
print(dense1.output[:5])