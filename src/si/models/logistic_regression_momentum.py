import numpy as np

from si.data.dataset import Dataset
from si.models.logistic_regression import LogisticRegression
from si.statistics.sigmoid_function import sigmoid_function


class LogisticRegressionMomentum(LogisticRegression):

    """
    LogisticRegression with L2 regularization, trained with gradient descent with momentum.
    The update of each iteration accumulates the previous updates (velocity), which
    keeps the descent direction and speeds up convergence.
    """

    def __init__(self, 
                 l2_penalty: float = 1, 
                 alpha:      float = 0.001, 
                 max_iter:   int   = 1000, 
                 patience:   int   = 5, 
                 scale:      bool  = True, 
                 momentum:   float = 0.9, 
                 **kwargs):

        """
        Initializes the LogisticRegressionMomentum.

        Parameters
        ----------
        l2_penalty: float
            The L2 regularization parameter
        alpha: float
            The learning rate
        max_iter: int
            The maximum number of iterations
        patience: int
            The maximum number of iterations without improvement allowed
        scale: bool
            Whether to scale the data or not
        momentum: float
            The momentum coefficient (0 = standard gradient descent)
        """

        super().__init__(l2_penalty = l2_penalty, 
                         alpha      = alpha, 
                         max_iter   = max_iter, 
                         patience   = patience, 
                         scale      = scale, 
                         **kwargs)
        self.momentum = momentum

    def _fit(self, 
             dataset: Dataset) -> 'LogisticRegressionMomentum':

        """
        Estimates the theta and theta_zero coefficients, mean, std and cost_history
        using gradient descent with momentum.

        Parameters
        ----------
        dataset: Dataset
            The training dataset

        Returns
        -------
        self: LogisticRegressionMomentum
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

        velocity      = np.zeros(n)
        velocity_zero = 0

        i              = 0
        early_stopping = 0

        while i < self.max_iter and early_stopping < self.patience:

            y_pred = sigmoid_function(X.dot(self.theta) + self.theta_zero)

            gradient      = (1 / m) * (y_pred - dataset.y).dot(X)
            gradient_zero = (1 / m) * np.sum(y_pred - dataset.y)

            # momentum: v = momentum * v + alpha * gradient
            velocity      = self.momentum * velocity + self.alpha * gradient
            velocity_zero = self.momentum * velocity_zero + self.alpha * gradient_zero

            penalization_term = self.theta * (1 - self.alpha * (self.l2_penalty / m))

            self.theta      = penalization_term - velocity
            self.theta_zero = self.theta_zero - velocity_zero

            self.cost_history[i] = self.cost(dataset)

            if i > 0 and self.cost_history[i] >= self.cost_history[i - 1]:
                early_stopping += 1
            else:
                early_stopping = 0

            i += 1

        return self