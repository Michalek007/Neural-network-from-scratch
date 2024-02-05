import random
import numpy as np
import matplotlib.pyplot as plt

from network import NeuralNetwork
from layer import Layer
from loss import LossCategoricalCrossEntropy, LossMSE
from activation import ActivationSoftmax, ActivationReLU, ActivationTanh
from nn_utils import one_hot_encoder
from app import app
from database.schemas import DigitImages


def xor_tanh():
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


def mnist_tanh():
    # from keras.datasets import mnist
    # from keras.utils import to_categorical
    #
    # # load MNIST from server
    # (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # #
    # # # training data : 60000 samples
    # # # reshape and normalize input data
    # x_train = x_train.reshape(x_train.shape[0], 1, 28*28)
    # x_train = x_train.astype('float32')
    # x_train /= 255
    # # encode output which is a number in range [0,9] into a vector of size 10
    # # e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    # y_train = to_categorical(y_train)
    #
    # # same for test data : 10000 samples
    # x_test = x_test.reshape(x_test.shape[0], 1, 28*28)
    # x_test = x_test.astype('float32')
    # x_test /= 255
    # y_test = to_categorical(y_test)

    with app.app_context():
        image_list = DigitImages.query.all()

    random.shuffle(image_list)
    x, y = [image.path for image in image_list], [image.digit for image in image_list]

    x, y = x[0:1005], y[0:1005]

    x = np.array([plt.imread(path) for path in x])

    x = x.reshape(x.shape[0], 1, 28 * 28)
    x = x.astype('float32')
    x /= 255

    y = one_hot_encoder(y, 10)
    y = np.array(y)

    # x = np.array([np.array([i]) for i in x])
    # y = np.array([np.array([i]) for i in y])

    x_train = x[0:1000]
    y_train = y[0:1000]

    x_test = x[1001:1005]
    y_test = y[1001:1005]

    network = NeuralNetwork()
    network.add_layer(layer=Layer(28 * 28, 100))
    network.add_layer(layer=ActivationTanh())
    network.add_layer(layer=Layer(100, 50))
    network.add_layer(layer=ActivationTanh())
    network.add_layer(layer=Layer(50, 10))
    network.add_layer(layer=ActivationTanh())
    network.loss_function = LossMSE()

    network.fit(x_train, y_train, epochs=35, learning_rate=0.1)

    out = network.predict(x_test)
    print('\n')
    print('Predicted values : ')
    print(out, end='\n')
    print('True values : ')
    print(y_test)

    network.save_model('models\\mnist_model_2.json')


if __name__ == '__main__':
    # xor_tanh()
    mnist_tanh()
