import numpy as np

from si.base.transformer import Transformer
from si.data.dataset import Dataset


class VarianceThreshold(Transformer):

    """
    Feature selector that removes all features whose variance is below a threshold.
    """

    def __init__(self, 
                 threshold: float = 0.0, 
                 **kwargs):

        """
        Initializes the VarianceThreshold.

        Parameters
        ----------
        threshold: float
            Cutoff value; features with variance not greater than this are removed
        """

        super().__init__(**kwargs)
        self.threshold = threshold
        self.variance  = None

    def _fit(self, 
             dataset: Dataset) -> 'VarianceThreshold':

        """
        Estimates the variance of each feature.

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit

        Returns
        -------
        self: VarianceThreshold
            The fitted transformer
        """

        self.variance = np.var(dataset.X, axis = 0)
        return self

    def _transform(self, 
                   dataset: Dataset) -> Dataset:

        """
        Selects all features with variance greater than the threshold.

        Parameters
        ----------
        dataset: Dataset
            The dataset to transform

        Returns
        -------
        Dataset
            The transformed dataset, with only the selected features
        """

        mask = self.variance > self.threshold

        X = dataset.X[:, mask]
        features = np.array(dataset.features)[mask].tolist()

        return Dataset(X, 
                       dataset.y, 
                       features = features, 
                       label    = dataset.label)


if __name__ == '__main__':
    dataset_ = Dataset.from_random(100, 10, 2)

    selector = VarianceThreshold(threshold = 0.08)
    selector.fit(dataset_)
    new_dataset = selector.transform(dataset_)
    print(new_dataset.shape())