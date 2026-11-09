from abc import abstractmethod

import numpy as np


class LossFunction:

    """
    Base class for the loss functions of a neural network.
    """

    @abstractmethod
    def loss(self, 
             y_true: np.ndarray, 
             y_pred: np.ndarray) -> float:

        """
        Computes the loss based on the true and predicted labels.
        Abstract method that needs to be implemented by all subclasses.

        Parameters
        ----------
        y_true: np.ndarray
            The true labels
        y_pred: np.ndarray
            The predicted labels
        """

        raise NotImplementedError

    @abstractmethod
    def derivative(self, 
                   y_true: np.ndarray, 
                   y_pred: np.ndarray) -> np.ndarray:

        """
        Computes the derivative of the loss based on the true and predicted labels.
        Abstract method that needs to be implemented by all subclasses.

        Parameters
        ----------
        y_true: np.ndarray
            The true labels
        y_pred: np.ndarray
            The predicted labels
        """

        raise NotImplementedError


class MeanSquaredError(LossFunction):

    """
    Mean squared error loss function, for regression tasks.
    """

    def loss(self, 
             y_true: np.ndarray, 
             y_pred: np.ndarray) -> float:

        """
        Computes the mean squared error: L = 1/n * sum (y_i - y*_i)^2.

        Parameters
        ----------
        y_true: np.ndarray
            The true labels
        y_pred: np.ndarray
            The predicted labels

        Returns
        -------
        loss: float
            The mean squared error
        """

        return np.mean((y_true - y_pred) ** 2)

    def derivative(self, 
                   y_true: np.ndarray, 
                   y_pred: np.ndarray) -> np.ndarray:

        """
        Computes the derivative of the mean squared error: dL/dY* = 2/n * (Y* - Y).

        Parameters
        ----------
        y_true: np.ndarray
            The true labels
        y_pred: np.ndarray
            The predicted labels

        Returns
        -------
        derivative: np.ndarray
            The derivative of the mean squared error
        """

        return 2 * (y_pred - y_true) / y_true.size


class BinaryCrossEntropy(LossFunction):

    """
    Binary cross-entropy loss function, for binary classification tasks.
    """

    def loss(self, 
             y_true: np.ndarray, 
             y_pred: np.ndarray) -> float:

        """
        Computes the binary cross-entropy: L = - sum (y_i log(y*_i) + (1 - y_i) log(1 - y*_i)).

        Parameters
        ----------
        y_true: np.ndarray
            The true labels
        y_pred: np.ndarray
            The predicted probabilities

        Returns
        -------
        loss: float
            The binary cross-entropy
        """

        p = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.sum(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))

    def derivative(self, 
                   y_true: np.ndarray, 
                   y_pred: np.ndarray) -> np.ndarray:

        """
        Computes the derivative of the binary cross-entropy: dL/dY* = -Y/Y* + (1 - Y)/(1 - Y*).

        Parameters
        ----------
        y_true: np.ndarray
            The true labels
        y_pred: np.ndarray
            The predicted probabilities

        Returns
        -------
        derivative: np.ndarray
            The derivative of the binary cross-entropy
        """

        p = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -(y_true / p) + (1 - y_true) / (1 - p)

class CategoricalCrossEntropy(LossFunction):

    """
    Categorical cross-entropy loss function, for multi-class classification tasks
    (true labels one-hot encoded).
    """

    def loss(self, 
             y_true: np.ndarray, 
             y_pred: np.ndarray) -> float:

        """
        Computes the categorical cross-entropy: L = - sum y_i log(y*_i).

        Parameters
        ----------
        y_true: np.ndarray
            The true labels (one-hot encoded)
        y_pred: np.ndarray
            The predicted class probabilities

        Returns
        -------
        loss: float
            The categorical cross-entropy
        """

        p = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.sum(y_true * np.log(p))

    def derivative(self, 
                   y_true: np.ndarray, 
                   y_pred: np.ndarray) -> np.ndarray:

        """
        Computes the derivative of the categorical cross-entropy: dL/dY* = -Y/Y*.

        Parameters
        ----------
        y_true: np.ndarray
            The true labels (one-hot encoded)
        y_pred: np.ndarray
            The predicted class probabilities

        Returns
        -------
        derivative: np.ndarray
            The derivative of the categorical cross-entropy
        """


        p = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -y_true / p