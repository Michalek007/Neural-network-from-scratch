import numpy as np

from .activation_base import Activation


class ActivationTanh(Activation):
    """ Hyperbolic tangent -> y = tanh(x)

        Derivative -> y' = 1 - tanh(x) ^ 2
    """

    def forward(self, inputs):
        self.input = inputs
        self.output = np.tanh(inputs)

    def backward(self, output_error, learning_rate=None):
        self.input_error = (1 - np.tanh(self.input) ** 2) * output_error
