import numpy as np


def one_hot_encoder(input_array, n_classes: int):
    """ Encodes input array in one hot encoding.
        Args:
            input_array: array which consist integers from 0 to N
            n_classes: number of classes, will be length of encoded one hot vector
        Returns:
            One hot encoded array.
    """
    output_array = []
    N = n_classes
    for i in range(len(input_array)):
        i_array = [0 for _ in range(N)]
        i_array[input_array[i]] = 1
        output_array.append(i_array)
    return output_array
