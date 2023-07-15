import numpy as np

test = [
    [1,0,0],
    [1,0,0],
    [0,1,0],
    [0,1,0]
]

# axis 0 is the rows, (1 + 1 + 0 + 0) = 2 = theSum[0]

# axis 1 is the columns (1 + 0 + 0) = theSum[0]


print('test', test)
print('sum(test, axis=0)', np.sum(test, axis=0))
print('sum(test, axis=1)', np.sum(test, axis=1))


layer_outputs = [
    [4.8, 1.21, 2.385],
    [8.9, -1.81, 0.2],
]
 
print('layer_outputs', layer_outputs)

# in axis 0
#    [4.8, 1.21, 2.385],
#    [8.9, -1.81, 0.2],
#    ____|______|_____|
#      1     0     0

print('np.argmax(layer_outputs, axis=0)', np.argmax(layer_outputs, axis=0)) # [1, 0, 0]


# in axis 0
#    [4.8, 1.21, 2.385], | 0
#    [8.9, -1.81, 0.2],  | 0
print('np.argmax(layer_outputs, axis=1)', np.argmax(layer_outputs, axis=1)) # [0, 0]