import numpy as np

a = [1, 2, 3]
b = [2, 3, 4]

a = np.array([a]) # (1,3)
b = np.array([b]) # (1,3)

print('row vector a', a)
print('column vector a.T', a.T)
print('b', b)

print('np.dot(a, b.T)', np.dot(a, b.T))

print('np.dot(a.T, b)', np.dot(a.T, b))