import numpy as np

from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy
from si.statistics.sigmoid_function import sigmoid_function


class LogisticRegression(Model):

    """
    The LogisticRegression is a logistic model using the L2 regularization.
    This model solves the logistic regression problem using an adapted Gradient Descent technique.
    """

    def __init__(self, 
                 l2_penalty: float = 1, 
                 alpha:      float = 0.001, 
                 max_iter:   int   = 1000, 
                 patience:   int   = 5, 
                 scale:      bool  = True, 
                 **kwargs):

        """
        Initializes the LogisticRegression.

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
             dataset: Dataset) -> 'LogisticRegression':

        """
        Estimates the theta and theta_zero coefficients, mean, std and cost_history
        using gradient descent.

        Parameters
        ----------
        dataset: Dataset
            The training dataset

        Returns
        -------
        self: LogisticRegression
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

            penalization_term = self.theta * (1 - self.alpha * (self.l2_penalty / m))

            self.theta = penalization_term - gradient

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
        Predicts the dependent variable (y) using theta, theta_zero and the sigmoid function,
        and converts the probabilities to binary values (prob < 0.5 -> 0, prob >= 0.5 -> 1).

        Parameters
        ----------
        dataset: Dataset
            The dataset to predict

        Returns
        -------
        predictions: np.ndarray
            The predicted classes (0 or 1)
        """

        X = self._scale_X(dataset.X)
        probabilities = sigmoid_function(X.dot(self.theta) + self.theta_zero)

        return np.where(probabilities >= 0.5, 1, 0)

    def _score(self, 
               dataset:     Dataset, 
               predictions: np.ndarray) -> float:

        """
        Computes the accuracy between the real and predicted y values.

        Parameters
        ----------
        dataset: Dataset
            The test dataset
        predictions: np.ndarray
            The predicted classes

        Returns
        -------
        accuracy: float
            The accuracy of the model on the dataset
        """

        return accuracy(dataset.y, predictions)

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

        X = self._scale_X(dataset.X)
        m = dataset.shape()[0]

        y_pred = sigmoid_function(X.dot(self.theta) + self.theta_zero)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

        cross_entropy  = -np.sum(dataset.y * np.log(y_pred) + (1 - dataset.y) * np.log(1 - y_pred)) / m
        regularization = (self.l2_penalty / (2 * m)) * np.sum(self.theta ** 2)

        return cross_entropy + regularization


if __name__ == '__main__':
    from si.model_selection.split import train_test_split

    dataset_                    = Dataset.from_random(600, 100, 2)
    dataset_train, dataset_test = train_test_split(dataset_, 
                                                   test_size    = 0.2, 
                                                   random_state = 42)

    model = LogisticRegression(l2_penalty = 1, alpha = 0.001, max_iter = 1000)
    model.fit(dataset_train)

    print(f'Iterations: {len(model.cost_history)}')
    print(f'Score:      {model.score(dataset_test)}')
    print(f'Cost:       {model.cost(dataset_test)}')