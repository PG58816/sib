from typing import Callable, Dict

import numpy as np

from si.data.dataset import Dataset
from si.model_selection.grid_search import cross_validate_configuration, get_folds


def log_uniform(low:  float, 
                high: float) -> Callable:

    """
    Returns a function that samples values uniformly on a log scale between low and high:
    10 ** uniform(log10(low), log10(high)).

    Parameters
    ----------
    low: float
        The lower bound
    high: float
        The upper bound

    Returns
    -------
    sampler: Callable
        A function that receives a random generator and returns one sampled value
    """

    def sampler(rng: np.random.Generator) -> float:
        return float(10 ** rng.uniform(np.log10(low), np.log10(high)))

    return sampler


def randomized_search_cv(model, 
                         dataset:                      Dataset, 
                         hyperparameter_distributions: Dict, 
                         scoring:                      Callable = None, 
                         cv:                           int      = 5, 
                         n_iter:                       int      = 10, 
                         seed:                         int      = None) -> Dict:

    """
    Randomized search with k-fold cross validation: evaluates n_iter random configurations
    and selects the one with the highest mean validation score.

    Parameters
    ----------
    model
        The model to tune, or a builder function that returns a new model
    dataset: Dataset
        The training dataset (never the test set)
    hyperparameter_distributions: Dict
        Dictionary {name: list of values or sampler function}
    scoring: Callable
        Scoring function scoring(y_true, y_pred) (if None, uses model.score)
    cv: int
        The number of folds
    n_iter: int
        The number of random configurations to evaluate
    seed: int
        The seed used to create the folds and to sample the configurations

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

    for _ in range(n_iter):

        configuration = {}
        for name, distribution in hyperparameter_distributions.items():
            if callable(distribution):
                configuration[name] = distribution(rng)
            else:
                configuration[name] = distribution[rng.integers(len(distribution))]

        score = cross_validate_configuration(model, dataset, configuration, folds, scoring)

        results['hyperparameters'].append(configuration)
        results['scores'].append(score)

    best_idx = int(np.argmax(results['scores']))
    results['best_hyperparameters'] = results['hyperparameters'][best_idx]
    results['best_score']           = results['scores'][best_idx]

    return results