import copy
import itertools
from typing import Callable, Dict, List

import numpy as np

from si.data.dataset import Dataset


def get_folds(n_samples: int, 
              cv:        int, 
              rng:       np.random.Generator) -> List[np.ndarray]:

    """
    Splits the indices of the samples into cv folds (after a random permutation).

    Parameters
    ----------
    n_samples: int
        The number of samples
    cv: int
        The number of folds
    rng: np.random.Generator
        The random generator used to permute the samples

    Returns
    -------
    folds: List[np.ndarray]
        The indices of each fold
    """

    indices = rng.permutation(n_samples)
    return np.array_split(indices, cv)


def build_model(model, 
                hyperparameters: Dict):

    """
    Builds a fresh, untrained model with the given hyperparameters.
    If model is a builder function, it is called with the hyperparameters;
    otherwise the model is copied and its attributes are changed.

    Parameters
    ----------
    model
        The model to tune, or a builder function that returns a new model
    hyperparameters: Dict
        The hyperparameters {name: value}

    Returns
    -------
    model
        A fresh model with the given hyperparameters
    """

    if callable(model) and not hasattr(model, 'fit'):
        return model(**hyperparameters)

    new_model = copy.deepcopy(model)
    for name, value in hyperparameters.items():
        setattr(new_model, name, value)
    return new_model


def cross_validate_configuration(model, 
                                 dataset:         Dataset, 
                                 hyperparameters: Dict, 
                                 folds:           List[np.ndarray], 
                                 scoring:         Callable = None) -> float:

    """
    Cross validates one configuration: for each fold, builds a fresh model,
    fits it on the training folds and scores it on the validation fold.

    Parameters
    ----------
    model
        The model to tune, or a builder function that returns a new model
    dataset: Dataset
        The training dataset
    hyperparameters: Dict
        The configuration to evaluate {name: value}
    folds: List[np.ndarray]
        The indices of each fold
    scoring: Callable
        Scoring function scoring(y_true, y_pred) (if None, uses model.score)

    Returns
    -------
    score: float
        The mean of the validation scores of the k folds
    """

    scores = []

    for i in range(len(folds)):
        val_idxs   = folds[i]
        train_idxs = np.concatenate([folds[j] for j in range(len(folds)) if j != i])

        train = Dataset(dataset.X[train_idxs], dataset.y[train_idxs], 
                        features = dataset.features, label = dataset.label)
        val   = Dataset(dataset.X[val_idxs],   dataset.y[val_idxs], 
                        features = dataset.features, label = dataset.label)

        fold_model = build_model(model, hyperparameters)

        fold_model.fit(train)

        if scoring is None:
            scores.append(fold_model.score(val))
        else:
            scores.append(scoring(val.y, fold_model.predict(val)))

    return float(np.mean(scores))


def grid_search_cv(model, 
                   dataset:             Dataset, 
                   hyperparameter_grid: Dict[str, list], 
                   scoring:             Callable = None, 
                   cv:                  int      = 5, 
                   seed:                int      = None) -> Dict:

    """
    Grid search with k-fold cross validation: evaluates every combination of the
    hyperparameter values and selects the one with the highest mean validation score.

    Parameters
    ----------
    model
        The model to tune, or a builder function that returns a new model
    dataset: Dataset
        The training dataset (never the test set)
    hyperparameter_grid: Dict[str, list]
        Dictionary {name: list of values to try}
    scoring: Callable
        Scoring function scoring(y_true, y_pred) (if None, uses model.score)
    cv: int
        The number of folds
    seed: int
        The seed used to create the folds

    Returns
    -------
    results: Dict
        hyperparameters: list with every configuration tested
        scores: mean validation score of each configuration
        best_hyperparameters, best_score: the configuration with the highest mean score
    """

    rng   = np.random.default_rng(seed)
    folds = get_folds(dataset.shape()[0], cv, rng)

    results = {'hyperparameters': [], 'scores': []}

    names = list(hyperparameter_grid.keys())

    for values in itertools.product(*hyperparameter_grid.values()):
        configuration = dict(zip(names, values))
        score = cross_validate_configuration(model, dataset, configuration, folds, scoring)

        results['hyperparameters'].append(configuration)
        results['scores'].append(score)


    best_idx = int(np.argmax(results['scores']))
    results['best_hyperparameters'] = results['hyperparameters'][best_idx]
    results['best_score']           = results['scores'][best_idx]

    return results