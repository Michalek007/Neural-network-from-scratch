import numpy as np

from .loss_base import Loss


class LossCategoricalCrossEntropy(Loss):
    """ Categorical Cross Entropy -> L = sum( y_true * ln(y_pred) ) """

    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true

        samples = len(y_pred)
        y_pred = np.clip(y_pred, 1e-7, 1-1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred * y_true, axis=1)
        else:
            raise ValueError('Incorrect shape of y_true argument. Expected length of shape: 1 or 2')

        negative_log_likelihoods = -np.log(correct_confidences)
        self.output = negative_log_likelihoods

    def backward(self):
        """ y_pred -> calculated output of nn (input), y_true -> wanted output """
        self.input_error = self.y_pred - self.y_true
