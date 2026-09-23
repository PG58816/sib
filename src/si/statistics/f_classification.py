from typing import Tuple

import numpy as np
from scipy import stats

from si.data.dataset import Dataset


def f_classification(dataset: Dataset) -> Tuple[np.ndarray, np.ndarray]:

    """
    Performs a one-way ANOVA F-test on the dataset, scoring the discriminative
    power of each feature with respect to the class label.

    Parameters
    ----------
    dataset: Dataset
        The Dataset object

    Returns
    -------
    F: np.ndarray
        The F value for each feature
    p: np.ndarray
        The p-value for each feature
    """

    classes = dataset.get_classes()
    groups  = [dataset.X[dataset.y == c] for c in classes]

    F, p = stats.f_oneway(*groups)

    return F, p