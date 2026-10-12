import numpy as np


def sigmoid_function(X: np.ndarray) -> np.ndarray:

    """
    Computes the sigmoid function for the input values.

    Parameters
    ----------
    X: np.ndarray
        The input values

    Returns
    -------
    sigmoid: np.ndarray
        The probability of the values being 1 (sigmoid function)
    """

    return 1 / (1 + np.exp(-X))