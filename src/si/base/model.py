from abc import abstractmethod

import numpy as np

from si.base.estimator import Estimator
from si.data.dataset import Dataset

class Model(Estimator):

    """
    Abstract base class for models.
    A model is an object that can predict the target values of a Dataset object.
    """

    def __init__(self, 
                 **kwargs):
        
        """
        Initialize the model.
        """

        super().__init__(**kwargs)

    def predict(self, dataset: Dataset) -> np.ndarray:

        """
        Predict the target values of the dataset.
        The model needs to be fitted before calling this method.

        Parameters
        ----------
        dataset: Dataset
            The dataset to predict the target values of.

        Returns
        -------
        predictions: np.ndarray
            The predicted target values.
        """

        if not self.is_fitted():
            raise ValueError('Model needs to be fitted before calling predict()')
        return self._predict(dataset)

    @abstractmethod
    def _predict(self, dataset: Dataset) -> np.ndarray:

        """
        Predict the target values of the dataset.
        Abstract method that needs to be implemented by all subclasses.

        Parameters
        ----------
        dataset: Dataset
            The dataset to predict the target values of.

        Returns
        -------
        predictions: np.ndarray
            The predicted target values.
        """

    def fit_predict(self, dataset: Dataset) -> np.ndarray:

        """
        Fit the model to the dataset and predict the target values.
        Equivalent to calling fit(dataset) and then predict(dataset).

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit and predict the target values of.

        Returns
        -------
        predictions: np.ndarray
            The predicted target values.
        """

        self.fit(dataset)
        return self.predict(dataset)

    def score(self, 
              dataset: 'Dataset') -> float:
        
        """
        Compute the score of the model on the dataset.
        The model needs to be fitted before calling this method.

        Parameters
        ----------
        dataset: Dataset
            The dataset to compute the score on.

        Returns
        -------
        score: float
            The score of the model on the dataset.
        """

        if not self.is_fitted():
            raise ValueError('Model needs to be fitted before calling score()')
        predictions = self._predict(dataset)
        return self._score(dataset, predictions)

    @abstractmethod
    def _score(self, 
               dataset: 'Dataset', predictions: np.ndarray) -> float:
        
        """
        Compute the score of the model on the dataset.
        Abstract method that needs to be implemented by all subclasses.

        Parameters
        ----------
        dataset: Dataset
            The dataset to compute the score on.
        predictions: np.ndarray
            The predictions of the model.

        Returns
        -------
        score: float
            The score of the model on the dataset.
        """
        
