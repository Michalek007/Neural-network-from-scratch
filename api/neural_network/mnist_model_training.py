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
from mnist_data_loader import MnistDataloader


def prepare_mnist_data(x, y):
    """ Prepares data obtained from MNIST dataset to be used in NeuralNetwork methods.
        Args:
            x: numpy array containing image data (pixels) in shape N X 28 x 28
            y: array containing numbers from 0 to 9
        Returns:
            x -> numpy array containing vectors of length 28*28
            y -> numpy array containing one hot encoded vectors
    """

    # reshaping 28x28 to vectors 28*28 and scaling by max value (255)
    x = x.reshape(x.shape[0], 1, 28 * 28)
    x = x.astype('float32')
    x /= 255

    # encoding data with one hot code
    y = one_hot_encoder(y, 10)
    y = np.array(y)
    return x, y


def mnist_model_fitting(model_name: str, dataset: str = 'db', samples: int = 1000):
    """ Fitting model with MNIST dataset which can be obtained from:
        - local database via SQL query
        - archive directory via MnistDataLoader
        Args:
            model_name: name of file to which model will be saved (in models directory)
            dataset: name of the datasets, expected: db, archive
            samples: number of samples used in fitting, test set is always 10_1000
        Returns:
            tuple of (calculated loss from fitting, calculated accuracy based on test set)
    """
    if dataset not in ('db', 'archive'):
        raise(ValueError('Wrong dataset. Expected: "db", "archive"'))

    # samples
    N = samples

    if dataset == 'db':
        with app.app_context():
            image_list = DigitImages.query.all()

        random.shuffle(image_list)
        x, y = [image.path for image in image_list], [image.digit for image in image_list]
        x, y = x[0:N + 10_000], y[0:N + 10_000]
        x = np.array([plt.imread(path) for path in x])

        x_train, y_train, x_test, y_test = x[0:N], y[0:N], x[N:-1], y[N:-1]

    else:
        mnis_dataset_base_path = 'C:\\Users\\Public\\Projects\\MachineLearning\\Datasets\\archive\\'
        data_loader = MnistDataloader(
            training_images_filepath=mnis_dataset_base_path + 'train-images.idx3-ubyte',
            training_labels_filepath=mnis_dataset_base_path + 'train-labels.idx1-ubyte',
            test_images_filepath=mnis_dataset_base_path + 't10k-images.idx3-ubyte',
            test_labels_filepath=mnis_dataset_base_path + 't10k-labels.idx1-ubyte',
        )
        (x_train, y_train), (x_test, y_test) = data_loader.load_data()

        x_train, y_train = x_train[0:N], y_train[0:N]

        x_train = np.array(x_train)
        x_test = np.array(x_test)

    x_train, y_train = prepare_mnist_data(x_train, y_train)
    x_test, y_test = prepare_mnist_data(x_test, y_test)

    # structure of neural network
    network = NeuralNetwork()
    network.add_layer(layer=Layer(28 * 28, 100))
    network.add_layer(layer=ActivationTanh())
    network.add_layer(layer=Layer(100, 50))
    network.add_layer(layer=ActivationTanh())
    network.add_layer(layer=Layer(50, 10))
    network.add_layer(layer=ActivationTanh())
    network.set_loss_function(loss_function=LossMSE())

    loss = network.fit(x_train, y_train, epochs=50, learning_rate=0.1)

    accuracy = network.validate(x_test, y_test)

    # displaying some example results of model
    n_predict = 5
    results = network.predict(x_test[0:n_predict])
    print('\nPredicted values : ')
    print(np.array(results))
    print('\nTrue values : ')
    print(y_test[0:n_predict])

    network.save_model(f'models\\{model_name}')

    return loss, accuracy


def mnist_model_validate(model_name: str):
    """ Validates mnist model with test data (10_000 samples) from archive dataset.
        Args:
            model_name: name of them mnist model file from "neural_network\\models\\" directory
        Returns:
            calculated accuracy
    """
    mnis_dataset_base_path = 'C:\\Users\\Public\\Projects\\MachineLearning\\Datasets\\archive\\'
    data_loader = MnistDataloader(
        training_images_filepath=mnis_dataset_base_path + 'train-images.idx3-ubyte',
        training_labels_filepath=mnis_dataset_base_path + 'train-labels.idx1-ubyte',
        test_images_filepath=mnis_dataset_base_path + 't10k-images.idx3-ubyte',
        test_labels_filepath=mnis_dataset_base_path + 't10k-labels.idx1-ubyte',
    )

    (_, _), (x_test, y_test) = data_loader.load_data()

    x_test = np.array(x_test)
    x_test, y_test = prepare_mnist_data(x_test, y_test)

    network = NeuralNetwork(from_file=f'models\\{model_name}')

    accuracy = network.validate(x_test, y_test)
    return accuracy


if __name__ == '__main__':
    # mnist_model_fitting('test.json', samples=1000, dataset='archive')
    mnist_model_validate('test.json')
