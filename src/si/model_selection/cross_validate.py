import copy
from typing import List

import numpy as np

from si.data.dataset import Dataset


def k_fold_cross_validation(model, 
                            dataset: Dataset, 
                            cv:      int = 5, 
                            seed:    int = None) -> List[float]:

    """
    Evaluates a model with k-fold cross validation: the dataset is split into cv folds;
    each fold is used once as the test set while the remaining folds are used for training.

    Parameters
    ----------
    model
        The model to evaluate (must have fit and score methods)
    dataset: Dataset
        The dataset to evaluate the model on
    cv: int
        The number of folds
    seed: int
        Seed for the permutation of the samples

    Returns
    -------
    scores: List[float]
        The score of the model on each fold
    """

    if seed is not None:
        np.random.seed(seed)

    n_samples = dataset.shape()[0]
    indices   = np.random.permutation(n_samples)
    folds     = np.array_split(indices, cv)

    scores = []

    for i in range(cv):
        test_idxs  = folds[i]
        train_idxs = np.concatenate([folds[j] for j in range(cv) if j != i])

        train = Dataset(dataset.X[train_idxs], dataset.y[train_idxs], 
                        features = dataset.features, label = dataset.label)
        test  = Dataset(dataset.X[test_idxs],  dataset.y[test_idxs], 
                        features = dataset.features, label = dataset.label)

        fold_model = copy.deepcopy(model)
        fold_model.fit(train)
        scores.append(fold_model.score(test))

    return scores