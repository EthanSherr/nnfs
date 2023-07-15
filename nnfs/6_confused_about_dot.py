import numpy as np
# (5 x 4)?
a = [[0.49, 0.97, 0.53, 0.05],
     [0.33, 0.65, 0.62, 0.51],
     [1.00, 0.38, 0.61, 0.41], 
     [0.74, 0.27, 0.64, 0.17],
     [0.36, 0.17, 0.96, 0.12]]
# (4 x 5)
b = [[0.79, 0.32, 0.68, 0.90, 0.77, 1],
     [0.18, 0.39, 0.12, 0.93, 0.09, 2],
     [0.87, 0.42, 0.60, 0.71, 0.12, 3],
     [0.45, 0.55, 0.40, 0.78, 0.81, 4]]

result = np.dot(a,b)
print('dot(a,b) = ', result)


rowVector = np.array([[1,2,3]])
print('rowVector', rowVector)

# Where np.expand_dims() adds a new dimension at the index of the axis.
print('expanded?', np.expand_dims([1,2,3], axis=0))