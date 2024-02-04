import numpy as np


def spiral_data(points: int, classes: int):
    """ Creates spiral data points.
        Args:
            points: number of points
            classes: number of classes
        Returns: tuple of spiral data points array and corresponding classes array
    """
    X = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points)  # radius
        t = np.linspace(class_number*4, (class_number+1)*4, points) + np.random.randn(points)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    return X, y


def one_hot_encoder(input_array):
    """ Encodes input array in one hot encoding.
        Args:
            input_array: array which consist integers from domain n: <0, N> and n is natural number
        Returns:
            One hot encoded array.
    """
    output_array = []
    input_array_unique = set(input_array.copy())
    N = len(input_array_unique)
    for i in range(len(input_array)):
        i_array = [0 for _ in range(N)]
        i_array[input_array[i]] = 1
        output_array.append(i_array)
    return output_array

