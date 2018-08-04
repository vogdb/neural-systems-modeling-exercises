import numpy as np


def laminate_array_to_matrix(array):
    n = len(array)
    matrix = np.zeros((n, n))
    for i in range(n):
        matrix[i, :] = np.roll(array, i)
    return matrix
