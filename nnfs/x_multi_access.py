import numpy as np

a = [
    [0.7, 0.1, 0.2],
    [0.1, 0.5, 0.4],
    [0.02, 0.9, 0.08]
]

# print('a[0,0]', a[0,0]) # WRONG!  2d list doesn't accept touple dum dums!

npa = np.array(a)
print('npa', npa)

print('npa[0,0]', npa[0,0])

someRange = range(3)
print('someRange', someRange)

print('npa[range(3), 0]', npa[range(3), 0]) # grab that first column!

lengthAxis0 = npa.shape[0]
print('lengthAxis0 = npa.shape[0] #', lengthAxis0)
print('npa[lengthAxis0, 0]', npa[range(lengthAxis0), 0]) # grab that first column!


