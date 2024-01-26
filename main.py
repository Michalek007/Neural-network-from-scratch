import numpy as np

from layer_dense import LayerDense
from activation_relu import ActivationReLU
from activation_softmax import ActivationSoftmax
from loss import LossCategoricalCrossEntropy


def spiral_data(points: int, classes: int):
    X = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points)  # radius
        t = np.linspace(class_number*4, (class_number+1)*4, points) + np.random.randn(points)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    return X, y


def layer_dense_example():
    X = [
        [1, 2, 3, 2.5],
        [2.0, 5.0, -1.0, 2.0],
        [-1.5, 2.7, 3.3, -0.8]
    ]

    layer1 = LayerDense(4, 5)
    layer2 = LayerDense(5, 2)

    layer1.forward(X)
    layer2.forward(layer1.output)

    print(layer2.output)


def activation_relu_example():
    X, y = spiral_data(points=100, classes=3)

    layer1 = LayerDense(2, 5)
    activation = ActivationReLU()

    layer1.forward(X)
    activation.forward(layer1.output)
    print(activation.output)


def activation_soft_max_example():
    X, y = spiral_data(points=100, classes=3)

    layer1 = LayerDense(2, 3)
    activation1 = ActivationReLU()

    layer2 = LayerDense(3, 3)
    activation2 = ActivationSoftmax()

    layer1.forward(X)
    activation1.forward(layer1.output)

    layer2.forward(activation1.output)
    activation2.forward(layer2.output)
    print(activation2.output)


def neural_network_example():
    X, y = spiral_data(points=100, classes=3)

    layer1 = LayerDense(2, 3)
    activation1 = ActivationReLU()

    layer2 = LayerDense(3, 3)
    activation2 = ActivationSoftmax()

    loss_function = LossCategoricalCrossEntropy()

    layer1.forward(X)
    activation1.forward(layer1.output)

    layer2.forward(activation1.output)
    activation2.forward(layer2.output)

    loss = loss_function.calculate(activation2.output, y)
    print(loss)


if __name__ == '__main__':
    # layer_dense_example()
    # activation_relu_example()
    # activation_soft_max_example()
    neural_network_example()
