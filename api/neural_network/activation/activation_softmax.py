import numpy as np

from .activation_base import Activation


class ActivationSoftmax(Activation):
    """ Softmax function -> y = e^x / sum(e^x)

        Derivative -> y' = - xi * xj if j != i else xj * ( 1 - xi)
    """

    def forward(self, inputs):
        self.input = inputs
        # exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        exp_values = np.exp(inputs)
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)  # pr values, should sum up to 1
        self.output = probabilities

    def backward(self, output_error, learning_rate=None):
        # def indicator_function(val1, val2):
        #     if val1 == val2:
        #         return 1
        #     else:
        #         return 0
        # _, n = self.input.shape
        # input_error = np.zeros((n, n))
        # for i in range(n):
        #     for j in range(n):
        #         input_error[i][j] = self.output[0][i] * (indicator_function(i, j) - self.output[0][j])
        # self.input_error = input_error[0] * output_error
        self.input_error = output_error
