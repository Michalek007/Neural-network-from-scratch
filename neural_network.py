import numpy as np


class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.loss_function = None

    def add_layer(self, layer):
        self.layers.append(layer)

    def predict(self, input_data):
        """ Predicts output for given input data. """
        # sample dimension first
        samples = len(input_data)
        result = []

        # run network over all samples
        for i in range(samples):
            # forward propagation
            inputs = np.array([input_data[i]])
            for layer in self.layers:
                layer.forward(inputs)
                inputs = layer.output
            result.append(inputs)

        return result

    def fit(self, x_train, y_train, epochs, learning_rate=0.1):
        """ Trains network with given train data. """
        samples = len(x_train)

        err = 0
        for i in range(epochs):
            for j in range(samples):
                inputs = np.array([x_train[j]])

                # forward propagation
                for layer in self.layers:
                    layer.forward(inputs)
                    inputs = layer.output

                print(self.layers[-1].output)
                self.loss_function.forward(self.layers[-1].output, np.array([y_train[j]]))
                err += self.loss_function.calculate()

                # backward propagation
                self.loss_function.backward()
                input_error = self.loss_function.input_error
                for layer in reversed(self.layers):
                    layer.backward(input_error, learning_rate)
                    input_error = layer.input_error

            # calculate average error on all samples
            err /= samples
            print('epoch %d/%d   error=%f' % (i + 1, epochs, err))
