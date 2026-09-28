import numpy as np

from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.mse import mse


class RidgeRegression(Model):

    """
    Linear regression model with L2 regularization (Ridge Regression), trained with gradient descent.
    """

    def __init__(self, 
                 l2_penalty: float = 1, 
                 alpha:      float = 0.001, 
                 max_iter:   int   = 1000, 
                 patience:   int   = 5, 
                 scale:      bool  = True, 
                 **kwargs):

        """
        Initializes the RidgeRegression.

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
        """

        super().__init__(**kwargs)
        self.l2_penalty = l2_penalty
        self.alpha      = alpha
        self.max_iter   = max_iter
        self.patience   = patience
        self.scale      = scale

        self.theta        = None
        self.theta_zero   = None
        self.mean         = None
        self.std          = None
        self.cost_history = {}

    def _scale_X(self, 
                 X: np.ndarray) -> np.ndarray:

        """
        Scales X with the mean and std estimated in _fit (if scale is True).

        Parameters
        ----------
        X: np.ndarray
            The feature matrix

        Returns
        -------
        X: np.ndarray
            The (scaled) feature matrix
        """

        if self.scale:
            return (X - self.mean) / self.std
        return X

    def _fit(self, 
             dataset: Dataset) -> 'RidgeRegression':

        """
        Estimates the theta and theta_zero coefficients, mean, std and cost_history
        using gradient descent.

        Parameters
        ----------
        dataset: Dataset
            The training dataset

        Returns
        -------
        self: RidgeRegression
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

            y_pred = X.dot(self.theta) + self.theta_zero

            gradient          = (self.alpha / m) * (y_pred - dataset.y).dot(X)
            penalization_term = self.theta * (1 - self.alpha * (self.l2_penalty / m))

            self.theta      = penalization_term - gradient
            self.theta_zero = self.theta_zero - (self.alpha / m) * np.sum(y_pred - dataset.y)

            self.cost_history[i] = self.cost(dataset)

            if i > 0 and self.cost_history[i] >= self.cost_history[i - 1]:
                early_stopping += 1
            else:
                early_stopping = 0

            i += 1

        return self

    def _predict(self, 
                 dataset: Dataset) -> np.ndarray:

        """
        Predicts the dependent variable (y) using the estimated theta coefficients.

        Parameters
        ----------
        dataset: Dataset
            The dataset to predict

        Returns
        -------
        predictions: np.ndarray
            The predicted values of y
        """

        X = self._scale_X(dataset.X)
        return X.dot(self.theta) + self.theta_zero

    def _score(self, 
               dataset:     Dataset, 
               predictions: np.ndarray) -> float:

        """
        Computes the MSE between the real and predicted y values.

        Parameters
        ----------
        dataset: Dataset
            The test dataset
        predictions: np.ndarray
            The predicted values

        Returns
        -------
        error: float
            The MSE of the model on the dataset
        """

        return mse(dataset.y, predictions)

    def cost(self, 
             dataset: Dataset) -> float:

        """
        Computes the cost function J between the real and predicted y values.

        Parameters
        ----------
        dataset: Dataset
            The dataset to compute the cost on

        Returns
        -------
        cost: float
            The value of the cost function J
        """

        y_pred = self._predict(dataset)
        m      = dataset.shape()[0]

        return (np.sum((y_pred - dataset.y) ** 2) + self.l2_penalty * np.sum(self.theta ** 2)) / (2 * m)


if __name__ == '__main__':
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3
    dataset_ = Dataset(X = X, y = y)

    model = RidgeRegression(l2_penalty = 0, alpha = 0.1, max_iter = 2000)
    model.fit(dataset_)

    print(f'Theta:      {model.theta}')
    print(f'Theta zero: {model.theta_zero}')
    print(f'Score:      {model.score(dataset_)}')
    print(f'Cost:       {model.cost(dataset_)}')
    print(f'Predict:    {model.predict(Dataset(X = np.array([[3, 5]])))}')