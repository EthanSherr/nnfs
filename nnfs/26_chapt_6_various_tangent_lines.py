import matplotlib.pyplot as plt
import numpy as np

def f(x):
    return 2*x**2

def approximateTangentLine(func, x):
    delta = 0.0001
    m = (func(x+delta) - func(x))/delta
    b = func(x) - m*x
    return (m, b)

x = np.arange(0, 5, 0.01)
y = f(x)

tangentX = np.array(range(0, 5))
tangents = approximateTangentLine(f, tangentX)

for m, b, xPosition in zip(tangents[0], tangents[1], tangentX):
    tangentDelta = 0.9
    xPlot = np.array([xPosition - tangentDelta, xPosition, xPosition + tangentDelta])
    plt.plot(xPlot, m*xPlot + b)

plt.plot(x, y)
plt.show()