from typing import Callable

import numpy as np

from si.base.transformer import Transformer
from si.data.dataset import Dataset
from si.statistics.f_classification import f_classification


class SelectPercentile(Transformer):

    """
    Feature selector that selects a given percentage of features with the
    highest scores according to a scoring function.
    """

    def __init__(self, 
                 score_func: Callable = f_classification, 
                 percentile: float    = 10, **kwargs):

        """
        Initializes the SelectPercentile.

        Parameters
        ----------
        score_func: Callable
            Variance analysis function that returns F and p values for each feature
        percentile: float
            Percentile of features to select (e.g., 10 for the top 10%)
        """

        super().__init__(**kwargs)
        self.score_func = score_func
        self.percentile = percentile
        self.F = None
        self.p = None

    def _fit(self, 
             dataset: Dataset) -> 'SelectPercentile':

        """
        Estimates the F and p values for each feature using the score_func.

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit

        Returns
        -------
        self: SelectPercentile
            The fitted transformer
        """

        self.F, self.p = self.score_func(dataset)
        return self

    def _transform(self, 
                   dataset: Dataset) -> Dataset:

        """
        Selects the top percentile of features based on their F-values,
        handling ties at the threshold to keep the exact number of features.

        Parameters
        ----------
        dataset: Dataset
            The dataset to transform

        Returns
        -------
        Dataset
            The transformed dataset, with only the selected features
        """

        n_features = len(self.F)
        n_select   = int(n_features * self.percentile / 100)

        idxs = np.argsort(-self.F, kind = 'stable')[:n_select]
        idxs = np.sort(idxs)

        X = dataset.X[:, idxs]
        features = np.array(dataset.features)[idxs].tolist()

        return Dataset(X, dataset.y, 
                       features = features, 
                       label    = dataset.label)


if __name__ == '__main__':
    dataset_ = Dataset.from_random(100, 10, 2)

    selector = SelectPercentile(percentile=40)
    selector.fit(dataset_)
    new_dataset = selector.transform(dataset_)
    print(new_dataset.shape())