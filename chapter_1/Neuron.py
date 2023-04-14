class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
    def compute(self, inputs):
        sum = self.bias
        for i in range(len(inputs)):
            sum += inputs[i] * self.weights[i]
        return self.activation(sum)
    def activation(self, value): 
        return value
