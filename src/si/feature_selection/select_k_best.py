from typing import Callable

import numpy as np

from si.base.transformer import Transformer
from si.data.dataset import Dataset
from si.statistics.f_classification import f_classification


class SelectKBest(Transformer):

    """
    Feature selector that selects the k features with the highest scores
    according to a scoring function.
    """

    def __init__(self, 
                 score_func: Callable = f_classification, 
                 k:          int      = 10, 
                 **kwargs):

        """
        Initializes the SelectKBest.

        Parameters
        ----------
        score_func: Callable
            Variance analysis function that returns F and p values for each feature
        k: int
            Number of features to select
        """

        super().__init__(**kwargs)
        self.score_func = score_func
        self.k = k
        self.F = None
        self.p = None

    def _fit(self, dataset: Dataset) -> 'SelectKBest':

        """
        Estimates the F and p values for each feature using the score_func.

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit

        Returns
        -------
        self: SelectKBest
            The fitted transformer
        """

        self.F, self.p = self.score_func(dataset)
        return self

    def _transform(self, 
                   dataset: Dataset) -> Dataset:
        
        """
        Selects the top k features with the highest F value.

        Parameters
        ----------
        dataset: Dataset
            The dataset to transform

        Returns
        -------
        Dataset
            The transformed dataset, with only the k selected features
        """

        idxs = np.argsort(-self.F)[:self.k]
        idxs = np.sort(idxs)

        X = dataset.X[:, idxs]
        features = np.array(dataset.features)[idxs].tolist()

        return Dataset(X, 
                       dataset.y, 
                       features = features, 
                       label    = dataset.label)


if __name__ == '__main__':
    dataset_ = Dataset.from_random(100, 10, 2)

    selector = SelectKBest(k=5)
    selector.fit(dataset_)
    new_dataset = selector.transform(dataset_)
    print(new_dataset.shape())