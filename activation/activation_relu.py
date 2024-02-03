import numpy as np

from .activation_base import Activation


class ActivationReLU(Activation):
    """ ReLU function -> y = x if x > 0 else 0 """

    def forward(self, inputs):
        """ y = x if x > 0 else 0 """
        self.input = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, output_error, learning_rate=None):
        """ output_error -> [ A0 ... An] n-> len(input)
            y = 1 if x > 0 else 0
        """
        inputs = self.input.copy()
        inputs[inputs > 0] = 1
        self.input_error = inputs * output_error
