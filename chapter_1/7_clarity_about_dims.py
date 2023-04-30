import numpy as np
# 3x4 (3 samples, 4 features)
inputs = [[1.0, 2.0, 3.0, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]]

# 3x4 (3 neurons, 4 feature-inputs)
weights = [[0.2, 0.8, -0.5, 1.0], 
           [0.5, -0.91, 0.26, -0.5], 
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3.0, 0.5]

# 3x4 X 4x3 = 3x3 
# 3x3 + (1,3) = 3x3?
# 3 outputs x 3 neuron
result = np.dot(inputs, np.array(weights).T) + biases
print('result', result)