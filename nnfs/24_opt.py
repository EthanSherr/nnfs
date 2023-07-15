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

print('X', X)