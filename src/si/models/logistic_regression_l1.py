import numpy as np

from si.data.dataset import Dataset
from si.models.logistic_regression import LogisticRegression
from si.statistics.sigmoid_function import sigmoid_function


class LogisticRegressionL1(LogisticRegression):

    """
    LogisticRegression with L1 regularization (instead of L2), trained with gradient descent.
    The L1 term pushes the coefficients of the less useful features towards zero.
    """

    def __init__(self, 
                 l1_penalty: float = 1, 
                 alpha:      float = 0.001, 
                 max_iter:   int   = 1000, 
                 patience:   int   = 5, 
                 scale:      bool  = True, 
                 **kwargs):

        """
        Initializes the LogisticRegressionL1.

        Parameters
        ----------
        l1_penalty: float
            The L1 regularization parameter
        alpha: float
            The learning rate
        max_iter: int
            The maximum number of iterations
        patience: int
            The maximum number of iterations without improvement allowed
        scale: bool
            Whether to scale the data or not
        """

        super().__init__(l2_penalty = 0, 
                         alpha      = alpha, 
                         max_iter   = max_iter, 
                         patience   = patience, 
                         scale      = scale, 
                         **kwargs)
        self.l1_penalty = l1_penalty

    def _fit(self, 
             dataset: Dataset) -> 'LogisticRegressionL1':

        """
        Estimates the theta and theta_zero coefficients, mean, std and cost_history
        using gradient descent with L1 regularization.

        Parameters
        ----------
        dataset: Dataset
            The training dataset

        Returns
        -------
        self: LogisticRegressionL1
            The fitted model
        """

        if self.scale:
            self.mean = np.nanmean(dataset.X, axis = 0)
            self.std  = np.nanstd(dataset.X, axis = 0)

        X = self._scale_X(dataset.X)
        m, n = dataset.shape()

        self.theta        = np.zeros(n)
        self.theta_zero   = 0
        self.cost_history = {}

        i              = 0
        early_stopping = 0

        while i < self.max_iter and early_stopping < self.patience:

            y_pred = sigmoid_function(X.dot(self.theta) + self.theta_zero)

            gradient = (self.alpha / m) * (y_pred - dataset.y).dot(X)

            # L1 regularization term (subgradient of |theta|)
            penalization_term = (self.alpha * self.l1_penalty / m) * np.sign(self.theta)

            self.theta      = self.theta - gradient - penalization_term
            self.theta_zero = self.theta_zero - (self.alpha / m) * np.sum(y_pred - dataset.y)

            self.cost_history[i] = self.cost(dataset)

            if i > 0 and self.cost_history[i] >= self.cost_history[i - 1]:
                early_stopping += 1
            else:
                early_stopping = 0

            i += 1

        return self

    def cost(self, 
             dataset: Dataset) -> float:

        """
        Computes the cost function J (cross-entropy + L1 term)
        between the real and predicted y values.

        Parameters
        ----------
        dataset: Dataset
            The dataset to compute the cost on

        Returns
        -------
        cost: float
            The value of the cost function J
        """

        X = self._scale_X(dataset.X)
        m = dataset.shape()[0]

        y_pred = sigmoid_function(X.dot(self.theta) + self.theta_zero)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

        cross_entropy = -np.sum(dataset.y * np.log(y_pred) + (1 - dataset.y) * np.log(1 - y_pred)) / m
        l1_term       = (self.l1_penalty / m) * np.sum(np.abs(self.theta))

        return cross_entropy + l1_term