import numpy as np
import json

from layer import Layer
from activation import ActivationSoftmax, ActivationReLU, ActivationTanh
from loss import LossCategoricalCrossEntropy, LossMSE


class NeuralNetwork:
    def __init__(self, from_file: str = None):
        self.layers = []
        self.loss_function = None

        if from_file:
            self.load_model(file=from_file)

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
            inputs = input_data[i]
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
                inputs = x_train[j]

                # forward propagation
                for layer in self.layers:
                    layer.forward(inputs)
                    inputs = layer.output

                # print(self.layers[-1].output)
                self.loss_function.forward(self.layers[-1].output, y_train[j])
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

    def save_model(self, file: str):
        json_layers = []
        for layer in self.layers:
            if hasattr(layer, 'weights') and hasattr(layer, 'biases'):
                layer_dict = dict(
                    type=layer.__class__.__name__,
                    weights=layer.weights.tolist(),
                    biases=layer.biases.tolist()
                )
                json_layers.append(layer_dict)
            else:
                json_layers.append(dict(
                    type=layer.__class__.__name__
                ))
        json_layers.append(dict(type=self.loss_function.__class__.__name__))
        json_model = json.dumps(json_layers)

        with open(file, 'w') as f:
            f.write(json_model)

    def load_model(self, file: str):
        with open(file, 'r') as f:
            json_model = json.loads(f.read())

        for layer in json_model:
            if layer.get('type') == 'Layer':
                self.add_layer(layer=Layer(0, 0, dict(
                    weights=np.array(layer.get('weights')),
                    biases=np.array(layer.get('biases'))
                    )
                ))
            elif layer.get('type') == 'ActivationTanh':
                self.add_layer(layer=ActivationTanh())
            elif layer.get('type') == 'ActivationReLU':
                self.add_layer(layer=ActivationReLU())
            elif layer.get('type') == 'ActivationSoftmax':
                self.add_layer(layer=ActivationSoftmax())
            else:
                if layer.get('type') == 'LossMSE':
                    self.loss_function = LossMSE()
                elif layer.get('type') == 'LossCategoricalCrossEntropy':
                    self.loss_function = LossCategoricalCrossEntropy()
