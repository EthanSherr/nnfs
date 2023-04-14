from Neuron import Neuron

inputs = [1,2,3]
weights = [0.2, 0.8, -0.5]
bias = 2


print('output is', Neuron(weights, bias).compute(inputs))