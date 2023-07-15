import numpy as np

class Activation_SoftMax:
    def forward(self, inputs):
        # e^(inputs ^ row max(inputs)) for each input, keeping dims! (subtracting keeps output in interval [0, 1], otherwise e^1000 is too big...etc)
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # normalize
        exp_sum = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = exp_sum

class Activation_SoftMaxCustom:
    def forward(self, input):
        E = 2.71828182846
        output = []
        T = 0
        for value in input:
            out = E ** value
            output.append(out)
            T += out

        for i, out in enumerate(output):
            output[i] /= T
            
        self.output = output


class Activation_SoftMax_One:
    def forward(self, input):
        exp_values = np.exp(input)
        exp_sum = sum(exp_values)
        norm_values = exp_values / exp_sum
        self.output = norm_values

