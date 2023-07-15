import numpy as np

class_targets = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 1, 0],
    [0, 1, 0],
]

print('is this 0, 1, 1? ', np.argmax(class_targets, axis=1))