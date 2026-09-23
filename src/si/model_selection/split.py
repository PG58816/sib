from typing import Tuple

import numpy as np

from si.data.dataset import Dataset

def train_test_split(dataset:      Dataset, 
                     test_size:    float = 0.2, 
                     random_state: int   = None) -> Tuple[Dataset, Dataset]:

    """
    Splits a Dataset object into train and test Dataset objects.

    Parameters
    ----------
    dataset: Dataset
        The Dataset object to split into training and testing data
    test_size: float
        The size of the testing Dataset (e.g., 0.2 for 20%)
    random_state: int
        Seed for generating permutations

    Returns
    -------
    train: Dataset
        The training Dataset object
    test: Dataset
        The testing Dataset object
    """

    if random_state is not None:
        np.random.seed(random_state)

    n_samples = dataset.shape()[0]
    n_test    = int(n_samples * test_size)

    permutations = np.random.permutation(n_samples)

    test_idxs  = permutations[:n_test]
    train_idxs = permutations[n_test:]

    train = Dataset(dataset.X[train_idxs], 
                    dataset.y[train_idxs] if dataset.y is not None else None,
                    features = dataset.features, 
                    label    = dataset.label)

    test = Dataset(dataset.X[test_idxs], 
                   dataset.y[test_idxs] if dataset.y is not None else None,
                   features = dataset.features, 
                   label    = dataset.label)

    return train, test


def stratified_train_test_split(dataset:      Dataset, 
                                test_size:    float = 0.2, 
                                random_state: int   = None) -> Tuple[Dataset, Dataset]:

    """
    Splits a Dataset object into train and test Dataset objects, preserving
    the class proportions of the original dataset (stratified split).

    Parameters
    ----------
    dataset: Dataset
        The Dataset object to split into training and testing data
    test_size: float
        The size of the testing Dataset (e.g., 0.2 for 20%)
    random_state: int
        Seed for generating permutations

    Returns
    -------
    train: Dataset
        The training Dataset object
    test: Dataset
        The testing Dataset object
    """
    
    if random_state is not None:
        np.random.seed(random_state)

    labels, counts = np.unique(dataset.y, return_counts=True)

    train_idxs = []
    test_idxs = []

    for label, count in zip(labels, counts):
        n_test_for_label = int(count * test_size)

        label_idxs = np.where(dataset.y == label)[0]
        np.random.shuffle(label_idxs)

        test_idxs.extend(label_idxs[:n_test_for_label])
        train_idxs.extend(label_idxs[n_test_for_label:])

    train_idxs = np.array(train_idxs, dtype = int)
    test_idxs  = np.array(test_idxs,  dtype = int)

    train = Dataset(dataset.X[train_idxs], 
                    dataset.y[train_idxs],
                    features = dataset.features, 
                    label    = dataset.label)

    test = Dataset(dataset.X[test_idxs], 
                   dataset.y[test_idxs],
                   features = dataset.features,
                   label    = dataset.label)

    return train, test