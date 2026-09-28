import numpy as np


def manhattan_distance(x: np.ndarray, 
                       y: np.ndarray) -> np.ndarray:

    """
    Computes the Manhattan distance between a single sample x and multiple samples y.

    Parameters
    ----------
    x: np.ndarray
        A single sample (n_features,)
    y: np.ndarray
        Multiple samples (n_samples, n_features)

    Returns
    -------
    distances: np.ndarray
        Array of distances between x and each sample in y (n_samples,)
    """

    return np.sum(np.abs(x - y), axis = 1)