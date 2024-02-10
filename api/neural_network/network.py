import numpy as np
import json

from neural_network.layer import Layer
from neural_network.activation import ActivationSoftmax, ActivationReLU, ActivationTanh
from neural_network.loss import LossCategoricalCrossEntropy, LossMSE
from neural_network.loss.loss_base import Loss


class NeuralNetwork:
    """ Implements neural network model.

        Api:
            fit -> trains model with train data

            validate -> validates model with test data

            predict -> predicts results based on given input

           save_model -> saves model to json file

        Attributes:
            layers: list of Layers or activation function objects
            loss_function: Loss object
    """
    def __init__(self, from_file: str = None):
        self.layers = []
        self.loss_function = None

        if from_file:
            self.load_model(file=from_file)

    def add_layer(self, layer):
        """ Adds layer to network.
            Args:
                layer: Layer or activation function object
        """
        self.layers.append(layer)

    def set_loss_function(self, loss_function: Loss):
        """ Sets loss_function attribute.
            Args:
                loss_function: Loss object
        """
        self.loss_function = loss_function

    def predict(self, input_data):
        """ Predicts output for given input data.
            Args:
                input_data: numpy array of input data
            Returns:
                list of predicted outputs
        """
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

    def validate(self, x_test, y_test):
        """ Validates network for given test dataset.
            Args:
                x_test: numpy array of X test data
                y_test: numpy array of Y test data (true results)
            Returns:
                accuracy for given dataset (correct predictions / samples)
        """
        samples = len(x_test)
        correct = 0

        # run network over all samples
        for i in range(samples):
            # forward propagation
            inputs = x_test[i]
            for layer in self.layers:
                layer.forward(inputs)
                inputs = layer.output

            out = self.layers[-1].output
            predicted_class = np.argmax(out)

            true_class = np.argmax(y_test[i])

            if predicted_class == true_class:
                correct += 1

        accuracy = correct/samples

        print('\nAccuracy: ')
        print(accuracy)
        return accuracy

    def fit(self, x_train, y_train, epochs: int, learning_rate=0.1):
        """ Trains network with given train data.
            Args:
                x_train: numpy array of X train data
                y_train: numpy array of Y train data (true results)
                epochs: number of complete pass of given training dataset
                learning_rate: "speed" of learning for model
            Returns:
                 average loss function error from all epochs
        """
        samples = len(x_train)

        err = 0
        for i in range(epochs):
            for j in range(samples):
                inputs = x_train[j]

                # forward propagation
                for layer in self.layers:
                    layer.forward(inputs)
                    inputs = layer.output

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
        return err

    def save_model(self, file: str):
        """ Saves current model to given file.
            Args:
                file: file name or path
        """
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
        """ Loads model from given file.
            Args:
                file: file name or path
        """
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
