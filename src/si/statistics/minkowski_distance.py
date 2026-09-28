import numpy as np


def minkowski_distance(x: np.ndarray, 
                       y: np.ndarray, 
                       p: float = 2) -> np.ndarray:

    """
    Computes the Minkowski distance of order p between a single sample x and multiple samples y.
    p = 1 is the Manhattan distance and p = 2 is the Euclidean distance.

    Parameters
    ----------
    x: np.ndarray
        A single sample (n_features,)
    y: np.ndarray
        Multiple samples (n_samples, n_features)
    p: float
        The order of the distance (p >= 1)

    Returns
    -------
    distances: np.ndarray
        Array of distances between x and each sample in y (n_samples,)
    """

    return np.sum(np.abs(x - y) ** p, axis = 1) ** (1 / p)