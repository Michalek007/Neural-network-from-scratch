import numpy as np

from neural_network import NeuralNetwork
from layer import Layer
from activation import ActivationSoftmax, ActivationReLU, ActivationTanh
from loss import LossCategoricalCrossEntropy, LossMSE
from utils import spiral_data, one_hot_encoder


def spiral_data_relu_nn():
    X, y = spiral_data(points=100, classes=3)
    y = np.array(one_hot_encoder(input_array=y))
    X = np.array([np.array([i]) for i in X])
    y = np.array([np.array([i]) for i in y])

    network = NeuralNetwork()
    network.add_layer(layer=Layer(2, 3))
    network.add_layer(layer=ActivationReLU())
    network.add_layer(layer=Layer(3, 3))
    network.add_layer(layer=ActivationSoftmax())
    network.loss_function = LossCategoricalCrossEntropy()

    network.fit(x_train=X, y_train=y, epochs=50, learning_rate=0.7)

    out = network.predict(X)
    print('\n')
    print('Predicted values : ')
    print(out, end='\n')
    print('True values : ')
    print(y)


def xor_relu_nn():
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
    X = np.array([np.array([i]) for i in X])
    y = np.array([np.array([i]) for i in y])

    network = NeuralNetwork()
    network.add_layer(layer=Layer(2, 3))
    network.add_layer(layer=ActivationReLU())
    network.add_layer(layer=Layer(3, 2))
    network.add_layer(layer=ActivationSoftmax())
    network.loss_function = LossCategoricalCrossEntropy()

    network.fit(x_train=X, y_train=y, epochs=35, learning_rate=0.1)

    out = network.predict(X)
    print('\n')
    print('Predicted values : ')
    print(out, end='\n')
    print('True values : ')
    print(y)


def xor_tanh_nn():
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
    X = np.array([np.array([i]) for i in X])
    y = np.array([np.array([i]) for i in y])

    network = NeuralNetwork()
    network.add_layer(layer=Layer(2, 3))
    network.add_layer(layer=ActivationTanh())
    network.add_layer(layer=Layer(3, 1))
    network.add_layer(layer=ActivationTanh())
    network.loss_function = LossMSE()

    network.fit(x_train=X, y_train=y, epochs=1000, learning_rate=0.05)

    out = network.predict(X)
    print('\n')
    print('Predicted values : ')
    print(out, end='\n')
    print('True values : ')
    print(y)


def mnist_tanh_nn():
    from keras.datasets import mnist
    from keras.utils import to_categorical

    # load MNIST from server
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    #
    # # training data : 60000 samples
    # # reshape and normalize input data
    x_train = x_train.reshape(x_train.shape[0], 1, 28*28)
    x_train = x_train.astype('float32')
    x_train /= 255
    # encode output which is a number in range [0,9] into a vector of size 10
    # e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    y_train = to_categorical(y_train)

    # same for test data : 10000 samples
    x_test = x_test.reshape(x_test.shape[0], 1, 28*28)
    x_test = x_test.astype('float32')
    x_test /= 255
    y_test = to_categorical(y_test)

    network = NeuralNetwork()
    network.add_layer(layer=Layer(28 * 28, 100))
    network.add_layer(layer=ActivationTanh())
    network.add_layer(layer=Layer(100, 50))
    network.add_layer(layer=ActivationTanh())
    network.add_layer(layer=Layer(50, 10))
    network.add_layer(layer=ActivationTanh())
    network.loss_function = LossMSE()

    network.fit(x_train[0:1000], y_train[0:1000], epochs=35, learning_rate=0.1)

    # test on 3 samples
    out = network.predict(x_test[0:3])
    print('\n')
    print('Predicted values : ')
    print(out, end='\n')
    print('True values : ')
    print(y_test[0:3])


def spiral_data_tanh_nn():
    x_train, y_train = spiral_data(points=100, classes=3)
    y_train = np.array(one_hot_encoder(input_array=y_train))
    x_train = np.array([np.array([i]) for i in x_train])
    y_train = np.array([np.array([i]) for i in y_train])

    network = NeuralNetwork()
    network.add_layer(layer=ActivationTanh())
    network.add_layer(layer=Layer(2, 5))
    network.add_layer(layer=Layer(5, 5))
    network.add_layer(layer=ActivationTanh())
    network.add_layer(layer=Layer(5, 3))
    network.add_layer(layer=ActivationTanh())
    network.loss_function = LossMSE()

    network.fit(x_train[0:290], y_train[0:290], epochs=35, learning_rate=0.5)

    # test on 3 samples
    out = network.predict(x_train[290:295])
    print('\n')
    print('Predicted values : ')
    print(out, end="\n")
    print('True values : ')
    print(y_train[290:295])


# spiral_data_relu_nn()
# xor_relu_nn()
# xor_tanh_nn()
# mnist_tanh_nn()
spiral_data_tanh_nn()
