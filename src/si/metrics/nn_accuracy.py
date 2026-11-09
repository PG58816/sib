import numpy as np

from si.metrics.accuracy import accuracy


def nn_accuracy(y_true: np.ndarray, 
                y_pred: np.ndarray) -> float:

    """
    Computes the accuracy for the outputs of a neural network.
    The network outputs probabilities, so they are first converted to classes:
    a column of probabilities (n_samples, 1) is converted to 0/1 with a 0.5 threshold,
    and a matrix of probabilities or one-hot labels (n_samples, n_classes) is converted
    to class indices with argmax. Then the usual accuracy is computed.

    Parameters
    ----------
    y_true: np.ndarray
        The real label values
    y_pred: np.ndarray
        The predicted values (output of the neural network)

    Returns
    -------
    accuracy: float
        The portion of well classified samples
    """

    def to_classes(y: np.ndarray) -> np.ndarray:
        y = np.asarray(y)
        if y.ndim == 2 and y.shape[1] == 1:
            return np.round(y[:, 0])
        if y.ndim == 2:
            return np.argmax(y, axis = 1)
        return y

    return accuracy(to_classes(y_true), to_classes(y_pred))