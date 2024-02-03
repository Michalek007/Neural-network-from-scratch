import numpy as np

from layer import Layer
from activation import ActivationReLU, ActivationSoftmax
from loss import LossCategoricalCrossEntropy
from utils import spiral_data


def layer_example():
    X = [
        [1, 2, 3, 2.5],
        [2.0, 5.0, -1.0, 2.0],
        [-1.5, 2.7, 3.3, -0.8]
    ]

    layer1 = Layer(4, 5)
    layer2 = Layer(5, 2)

    layer1.forward(X)
    layer2.forward(layer1.output)

    print(layer2.output)


def activation_relu_example():
    X, y = spiral_data(points=100, classes=3)

    layer1 = Layer(2, 5)
    activation = ActivationReLU()

    layer1.forward(X)
    activation.forward(layer1.output)
    print(activation.output)


def activation_soft_max_example():
    X, y = spiral_data(points=100, classes=3)

    layer1 = Layer(2, 3)
    activation1 = ActivationReLU()

    layer2 = Layer(3, 3)
    activation2 = ActivationSoftmax()

    layer1.forward(X)
    activation1.forward(layer1.output)

    layer2.forward(activation1.output)
    activation2.forward(layer2.output)
    print(activation2.output)


def backward_propagation_example():
    X, y = spiral_data(points=100, classes=3)

    layer1 = Layer(2, 3)  # layer 1 2->3
    activation1 = ActivationReLU()  # activation 1 RELU
    layer2 = Layer(3, 3)   # layer 2 3->3
    activation2 = ActivationSoftmax()  # activation 2 SOFTMAX
    loss_function = LossCategoricalCrossEntropy()  # loss function CATEGORICAL ENTROPY

    input_data = np.array([X[1]])
    y_pred = None
    y_true = np.array([1, 0, 0])

    # forward propagation
    print('Starting forward propagation...')
    layer1.forward(input_data)
    activation1.forward(layer1.output)

    layer2.forward(activation1.output)
    activation2.forward(layer2.output)

    y_pred = activation2.output
    print('Calculated output: ')
    print(y_pred)
    print('Wanted output: ')
    print(y_true)

    # loss_function.forward(calculated_output_data, wanted_output_data)
    loss_function.forward(y_pred, y_true)
    print("Loss: ")
    print(loss_function.calculate())

    print('\n_______________________\n')

    # backward propagation
    print('Starting backward propagation...')
    loss_function.backward()
    print('Loss function input error: ')
    print(loss_function.input_error)

    activation2.backward(loss_function.input_error)
    print('Activation 2 softmax input_error:')
    print(activation2.input_error)

    layer2.backward(activation2.input_error, 0.1)
    print('Layer2 input_error: ')
    print(layer2.input_error)

    activation1.backward(layer2.input_error)
    print('Activation1 input_error: ')
    print(activation1.input_error)

    layer1.backward(activation1.input_error, 0.1)
    print('Layer1 input_error: ')
    print(layer1.input_error)


if __name__ == '__main__':
    # layer_example()
    # activation_relu_example()
    # activation_soft_max_example()
    backward_propagation_example()
