import numpy as np

from .loss_base import Loss


class LossMSE(Loss):
    """ Mean squared error -> L = sum( (y_true - y_pred) ^ 2 )

        Derivative -> L' = 2/n * (y_pred - y_true)
    """

    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true

        self.output = np.power(y_true-y_pred, 2)

    def backward(self):
        self.input_error = 2*(self.y_pred-self.y_true)/self.y_true.size
