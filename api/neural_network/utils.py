import numpy as np


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
