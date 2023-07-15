import numpy as np
# (3,4)
inputs = [[1.0, 2.0, 3.0, 2.5], 
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]]
# (3,4)
weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]
# (3,4) x (4,3) = (3x3)
print('np.dot(np.array(inputs).T, np.array(weights))', np.dot(np.array(weights).T, np.array(inputs)) )

# print('np.dot(np.array(weights), np.array(inputs))', np.dot(weights, inputs))