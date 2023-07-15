import numpy as np

layer_outputs = [
    [4.8, 1.21, 2.385],
    [8.9, -1.81, 0.2],
    [1.41, 1.051, 0.026]
]
print('layer_outputs', layer_outputs)

print('\nSum without axis')
print(np.sum(layer_outputs))

print('\nthis will be same since default is none')
print(np.sum(layer_outputs, axis=None))

print('\nnp.sum(layer_outputs, axis=0) #This means to sum row-wise, along axis 0: 4.8+8.9+1.41 = 15.11')
print(np.sum(layer_outputs, axis=0))

print('\nnp.sum(layer_outputs, axis=1) #This means to sum column-wise, along axis 0: 4.8+1.21+2.385 = 8.395')
print(np.sum(layer_outputs, axis=1))

