import numpy as np


def euclidean_distance(x: np.ndarray, 
                       y: np.ndarray) -> np.ndarray:

    """
    Computes the euclidean distance between a single sample x and multiple samples y.

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

    return np.sqrt(np.sum((x - y) ** 2, axis = 1))