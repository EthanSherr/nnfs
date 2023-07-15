import matplotlib.pyplot as plt
import numpy as np

def f(x):
    return 2*x*x

# using numerical differentiation
def numericalDerivative(func, x):
    x1 = x
    x2 = x + 0.0001
    y1 = func(x1)
    y2 = func(x2)

    return (y2 - y1)/(x2 - x1)

x = np.arange(0, 5, 0.001)
y = f(x)

print(numericalDerivative(f, 1))

plt.plot(x, f(x))
plt.plot(x, numericalDerivative(f, x))
plt.show()

