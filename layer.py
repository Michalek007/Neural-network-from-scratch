import numpy as np

np.random.seed(0)


class Layer:
    def __init__(self, n_inputs: int, n_neurons: int, from_dict: dict = None):

        if from_dict:
            self.weights = np.array(
                from_dict.get('weights')
            )
            self.biases = np.array(
                from_dict.get('biases')
            )

        else:
            self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
            self.biases = np.zeros((1, n_neurons))

        self.output = None
        self.input = None
        self.input_error = None

    def forward(self, inputs):
        self.input = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, output_error, learning_rate):
        self.input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        #  biases_error = output_error

        self.weights -= learning_rate * weights_error
        self.biases -= learning_rate * output_error
