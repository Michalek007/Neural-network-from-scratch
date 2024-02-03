import numpy as np
from abc import ABC, abstractmethod


class Loss(ABC):
    """ Base class for all loss functions.
        Attributes:
            output: calculated loss array
            input_error: error based on input
            y_pred: predicted y value (should be passed from activation function for last layer)
            y_true: true y values
    """
    def __init__(self):
        self.output = None
        self.input_error = None

        self.y_pred = None
        self.y_true = None

    def calculate(self):
        """ Calculates mean value of loss. """
        if self.output is None:
            return None
        mean_loss = np.mean(self.output)
        return mean_loss

    @abstractmethod
    def forward(self, y_pred, y_true):
        pass
