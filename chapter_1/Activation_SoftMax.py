import numpy as np

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

class Activation_SoftMax:
    def forward(self, inputs):
        exp_values = np.exp(inputs)
        exp_sum = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = exp_sum
