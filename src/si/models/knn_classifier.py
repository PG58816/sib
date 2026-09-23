from collections import Counter
from typing import Callable

import numpy as np

from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy
from si.statistics.euclidean_distance import euclidean_distance

class KNNClassifier(Model):

    """
    K-Nearest Neighbors classifier.
    Predicts the class of a sample based on the k most similar training examples.
    """

    def __init__(self, 
                 k: int = 5, 
                 distance: Callable = euclidean_distance, 
                 **kwargs):

        """
        Initializes the KNNClassifier.

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
             dataset: Dataset) -> 'KNNClassifier':

        """
        Stores the training dataset.

        Parameters
        ----------
        dataset: Dataset
            The training dataset

        Returns
        -------
        self: KNNClassifier
            The fitted classifier
        """

        self.dataset = dataset
        return self

    def _get_closest_label(self, 
                           sample: np.ndarray):

        """
        Returns the most common class among the k nearest neighbors of a sample.

        Parameters
        ----------
        sample: np.ndarray
            A single sample

        Returns
        -------
        label
            The predicted class for the sample
        """

        distances = self.distance(sample, self.dataset.X)

        k_nearest_neighbors        = np.argsort(distances)[:self.k]
        k_nearest_neighbors_labels = self.dataset.y[k_nearest_neighbors]

        counts      = Counter(k_nearest_neighbors_labels)
        most_common = counts.most_common(1)[0][0]
        return most_common

    def _predict(self, 
                 dataset: Dataset) -> np.ndarray:

        """
        Predicts the classes of the test dataset.

        Parameters
        ----------
        dataset: Dataset
            The test dataset

        Returns
        -------
        predictions: np.ndarray
            The predicted classes for the testing dataset
        """

        return np.array([self._get_closest_label(sample) for sample in dataset.X])

    def _score(self, 
               dataset:     Dataset, 
               predictions: np.ndarray) -> float:
        
        """
        Computes the accuracy between predictions and actual values.

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

if __name__ == '__main__':
    from si.model_selection.split import train_test_split

    dataset_                    = Dataset.from_random(600, 100, 2)
    dataset_train, dataset_test = train_test_split(dataset_, 
                                                   test_size    = 0.2, 
                                                   random_state = 42)

    knn   = KNNClassifier(k = 5)
    knn.fit(dataset_train)
    score = knn.score(dataset_test)
    print(f'Accuracy: {score}')