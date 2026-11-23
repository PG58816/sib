import os
from unittest import TestCase

import numpy as np

from datasets import DATASETS_PATH

from si.data.dataset import Dataset
from si.io.csv_file import read_csv
from si.model_selection.cross_validate import k_fold_cross_validation
from si.model_selection.grid_search import grid_search_cv
from si.model_selection.randomized_search import log_uniform, randomized_search_cv
from si.model_selection.split import stratified_train_test_split, train_test_split
from si.models.knn_classifier import KNNClassifier


class TestSplit(TestCase):

    def setUp(self):
        self.dataset = read_csv(os.path.join(DATASETS_PATH, 'iris', 'iris.csv'), features = True, label = True)

    def test_train_test_split(self):
        train, test = train_test_split(self.dataset, test_size = 0.2, random_state = 123)

        self.assertEqual(test.shape()[0], int(self.dataset.shape()[0] * 0.2))
        self.assertEqual(train.shape()[0] + test.shape()[0], self.dataset.shape()[0])
        self.assertEqual(train.features, self.dataset.features)

    def test_train_test_split_reproducible(self):
        train_1, _ = train_test_split(self.dataset, test_size = 0.2, random_state = 1)
        train_2, _ = train_test_split(self.dataset, test_size = 0.2, random_state = 1)
        np.testing.assert_array_equal(train_1.X, train_2.X)

    def test_stratified_train_test_split(self):
        train, test = stratified_train_test_split(self.dataset, test_size = 0.2, random_state = 123)

        self.assertEqual(train.shape(), (120, 4))
        self.assertEqual(test.shape(), (30, 4))

        # the class proportions are preserved
        _, train_counts = np.unique(train.y, return_counts = True)
        _, test_counts  = np.unique(test.y,  return_counts = True)
        np.testing.assert_array_equal(train_counts, [40, 40, 40])
        np.testing.assert_array_equal(test_counts,  [10, 10, 10])

    def test_stratified_no_overlap(self):
        X = np.arange(20).reshape(-1, 1).astype(float)
        y = np.array([0] * 10 + [1] * 10)
        train, test = stratified_train_test_split(Dataset(X, y), test_size = 0.3, random_state = 0)

        self.assertEqual(len(np.intersect1d(train.X.ravel(), test.X.ravel())), 0)
        self.assertEqual(train.shape()[0] + test.shape()[0], 20)


class TestKFoldCrossValidation(TestCase):

    def setUp(self):
        self.dataset = read_csv(os.path.join(DATASETS_PATH, 'iris', 'iris.csv'), features = True, label = True)

    def test_k_fold_cross_validation(self):
        model  = KNNClassifier(k = 3)
        scores = k_fold_cross_validation(model, self.dataset, cv = 5, seed = 42)

        self.assertEqual(len(scores), 5)
        self.assertTrue(all(0 <= s <= 1 for s in scores))
        self.assertGreater(np.mean(scores), 0.9)

        # the original model is not trained (a copy is used in each fold)
        self.assertFalse(model.is_fitted())


class TestGridSearch(TestCase):

    def setUp(self):
        self.dataset = read_csv(os.path.join(DATASETS_PATH, 'iris', 'iris.csv'), features = True, label = True)

    def test_grid_search_cv(self):
        grid    = {'k': [1, 3, 5]}
        results = grid_search_cv(KNNClassifier(), self.dataset, grid, cv = 3, seed = 42)

        self.assertEqual(results['hyperparameters'], [{'k': 1}, {'k': 3}, {'k': 5}])
        self.assertEqual(len(results['scores']), 3)
        self.assertIn(results['best_hyperparameters'], results['hyperparameters'])
        self.assertEqual(results['best_score'], max(results['scores']))

    def test_grid_search_combinations(self):
        calls = []

        def build_model(k, distance_name):
            calls.append((k, distance_name))
            return KNNClassifier(k = k)

        grid    = {'k': [1, 3], 'distance_name': ['a', 'b', 'c']}
        results = grid_search_cv(build_model, self.dataset, grid, cv = 2, seed = 1)

        # 2 x 3 configurations, each built once per fold
        self.assertEqual(len(results['hyperparameters']), 6)
        self.assertEqual(len(calls), 6 * 2)

    def test_grid_search_same_folds(self):
        # same seed -> same results
        grid = {'k': [1, 3]}
        r1 = grid_search_cv(KNNClassifier(), self.dataset, grid, cv = 3, seed = 7)
        r2 = grid_search_cv(KNNClassifier(), self.dataset, grid, cv = 3, seed = 7)
        self.assertEqual(r1['scores'], r2['scores'])


class TestRandomizedSearch(TestCase):

    def setUp(self):
        self.dataset = read_csv(os.path.join(DATASETS_PATH, 'iris', 'iris.csv'), features = True, label = True)

    def test_log_uniform(self):
        sampler = log_uniform(1e-4, 1e-1)
        rng     = np.random.default_rng(0)
        samples = np.array([sampler(rng) for _ in range(1000)])

        self.assertTrue(np.all((samples >= 1e-4) & (samples <= 1e-1)))

        # about one third of the samples in each order of magnitude
        for low in (1e-4, 1e-3, 1e-2):
            fraction = np.mean((samples >= low) & (samples < low * 10))
            self.assertGreater(fraction, 0.25)
            self.assertLess(fraction, 0.42)

    def test_randomized_search_cv(self):
        distributions = {'k': [1, 3, 5, 7]}
        results = randomized_search_cv(KNNClassifier(), self.dataset, distributions, cv = 3, n_iter = 5, seed = 42)

        self.assertEqual(len(results['hyperparameters']), 5)
        self.assertEqual(len(results['scores']), 5)
        self.assertTrue(all(h['k'] in [1, 3, 5, 7] for h in results['hyperparameters']))
        self.assertEqual(results['best_score'], max(results['scores']))

    def test_randomized_search_reproducible(self):
        distributions = {'k': [1, 3, 5, 7], 'scale': log_uniform(1e-2, 1e2)}

        def build_model(k, scale):
            return KNNClassifier(k = k)

        r1 = randomized_search_cv(build_model, self.dataset, distributions, cv = 3, n_iter = 4, seed = 3)
        r2 = randomized_search_cv(build_model, self.dataset, distributions, cv = 3, n_iter = 4, seed = 3)
        self.assertEqual(r1['hyperparameters'], r2['hyperparameters'])
        self.assertEqual(r1['scores'], r2['scores'])
