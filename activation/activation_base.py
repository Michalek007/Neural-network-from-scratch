from abc import ABC, abstractmethod


class Activation(ABC):
    """ Base class for all activation functions.
        Attributes:
            output: calculated output array (should be passed to next layer)
            input: given input array (output of previous layer)
            input_error: error based on input
    """
    def __init__(self):
        self.output = None
        self.input = None
        self.input_error = None

    @abstractmethod
    def forward(self, inputs):
        pass

    @abstractmethod
    def backward(self, output_error, learning_rate):
        pass
