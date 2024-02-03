import numpy as np

from neural_network import NeuralNetwork
from layer import Layer
from activation import ActivationSoftmax, ActivationReLU
from loss import LossCategoricalCrossEntropy
from utils import spiral_data, one_hot_encoder


# spiral data training
# X, y = spiral_data(points=100, classes=3)
# y = np.array(one_hot_encoder(input_array=y))
#
# layer1 = Layer(2, 3)
# activation1 = ActivationReLU()
# layer2 = Layer(3, 3)
# activation2 = ActivationSoftmax()
# loss_function = LossCategoricalCrossEntropy()
#
#
# network = NeuralNetwork()
# network.add_layer(layer=layer1)
# network.add_layer(layer=activation1)
# network.add_layer(layer=layer2)
# network.add_layer(layer=activation2)
# network.loss_function = loss_function
#
# network.fit(x_train=X, y_train=y, epochs=50)


# xor training
X = np.array([
    [0, 1],
    [1, 0],
    [0, 0],
    [1, 1]
])
y = np.array([
    1,
    1,
    0,
    0
])

layer1 = Layer(2, 4)
activation1 = ActivationReLU()
layer2 = Layer(4, 4)
activation2 = ActivationReLU()
layer3 = Layer(4, 2)
activation3 = ActivationSoftmax()
loss_function = LossCategoricalCrossEntropy()

network = NeuralNetwork()
network.add_layer(layer=layer1)
network.add_layer(layer=activation1)
network.add_layer(layer=layer2)
network.add_layer(layer=activation2)
network.add_layer(layer=layer3)
network.add_layer(layer=activation3)
network.loss_function = loss_function

network.fit(x_train=X, y_train=y, epochs=50)
