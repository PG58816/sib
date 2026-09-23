from typing import Callable

import numpy as np

from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.rmse import rmse
from si.statistics.euclidean_distance import euclidean_distance


class KNNRegressor(Model):

    """
    K-Nearest Neighbors regressor.
    Predicts the value of a sample as the average value of the k most similar training examples.
    """

    def __init__(self, 
                 k:        int      = 5, 
                 distance: Callable = euclidean_distance, 
                 **kwargs):

        """
        Initializes the KNNRegressor.

        Parameters
        ----------
        k: int
            The number of nearest neighbors to consider
        distance: Callable
            Function that calculates the distance between a sample and
            the samples in the training dataset
        """

        super().__init__(**kwargs)
        self.k        = k
        self.distance = distance
        self.dataset  = None

    def _fit(self, 
             dataset: Dataset) -> 'KNNRegressor':

        """
        Stores the training dataset.

        Parameters
        ----------
        dataset: Dataset
            The training dataset

        Returns
        -------
        self: KNNRegressor
            The fitted regressor
        """

        self.dataset = dataset
        return self

    def _get_closest_value(self, 
                           sample: np.ndarray) -> float:

        """
        Returns the average value of the k nearest neighbors of a sample.

        Parameters
        ----------
        sample: np.ndarray
            A single sample

        Returns
        -------
        value: float
            The predicted value for the sample
        """

        distances = self.distance(sample, self.dataset.X)

        k_nearest_neighbors        = np.argsort(distances)[:self.k]
        k_nearest_neighbors_values = self.dataset.y[k_nearest_neighbors]

        return np.mean(k_nearest_neighbors_values)

    def _predict(self, 
                 dataset: Dataset) -> np.ndarray:

        """
        Predicts the values of the test dataset.

        Parameters
        ----------
        dataset: Dataset
            The test dataset

        Returns
        -------
        predictions: np.ndarray
            The predicted values for the testing dataset
        """

        return np.array([self._get_closest_value(sample) for sample in dataset.X])

    def _score(self, 
               dataset:     Dataset, 
               predictions: np.ndarray) -> float:
        
        """
        Computes the RMSE between predictions and actual values.

        Parameters
        ----------
        dataset: Dataset
            The test dataset
        predictions: np.ndarray
            The predicted values

        Returns
        -------
        error: float
            The RMSE of the model on the dataset
        """

        return rmse(dataset.y, predictions)