import numpy as np

from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.mse import mse


class RidgeRegressionLeastSquares(Model):

    """
    Linear regression model with L2 regularization (Ridge Regression),
    solved in closed form with the least squares method.
    """

    def __init__(self, 
                 l2_penalty: float = 1, 
                 scale:      bool  = True, 
                 **kwargs):

        """
        Initializes the RidgeRegressionLeastSquares.

        Parameters
        ----------
        l2_penalty: float
            The L2 regularization parameter
        scale: bool
            Whether to scale the data or not
        """

        super().__init__(**kwargs)
        self.l2_penalty = l2_penalty
        self.scale      = scale

        self.theta      = None
        self.theta_zero = None
        self.mean       = None
        self.std        = None

    def _fit(self, 
             dataset: Dataset) -> 'RidgeRegressionLeastSquares':

        """
        Estimates the theta and theta_zero coefficients, mean and std
        using the closed-form least squares solution.

        Parameters
        ----------
        dataset: Dataset
            The training dataset

        Returns
        -------
        self: RidgeRegressionLeastSquares
            The fitted model
        """

        if self.scale:
            self.mean = np.nanmean(dataset.X, axis = 0)
            self.std  = np.nanstd(dataset.X, axis = 0)
            X = (dataset.X - self.mean) / self.std
        else:
            X = dataset.X

        X = np.c_[np.ones(X.shape[0]), X]

        penalty_matrix = self.l2_penalty * np.eye(X.shape[1])

        penalty_matrix[0, 0] = 0

        thetas = np.linalg.inv(X.T.dot(X) + penalty_matrix).dot(X.T).dot(dataset.y)

        self.theta_zero = thetas[0]
        self.theta      = thetas[1:]

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

        if self.scale:
            X = (dataset.X - self.mean) / self.std
        else:
            X = dataset.X

        X = np.c_[np.ones(X.shape[0]), X]

        thetas = np.r_[self.theta_zero, self.theta]

        return X.dot(thetas)

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


if __name__ == '__main__':
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3
    dataset_ = Dataset(X = X, y = y)

    model = RidgeRegressionLeastSquares(l2_penalty = 0, scale = True)
    model.fit(dataset_)

    print(f'Theta:      {model.theta}')
    print(f'Theta zero: {model.theta_zero}')
    print(f'Score:      {model.score(dataset_)}')
    print(f'Predict:    {model.predict(Dataset(X = np.array([[3, 5]])))}')