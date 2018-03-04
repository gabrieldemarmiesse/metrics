import numpy as np


def mse(arr1, arr2):
    return np.mean(np.square(arr1 - arr2))
