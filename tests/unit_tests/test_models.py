import os
from functools import partial
from unittest import TestCase

import numpy as np

from datasets import DATASETS_PATH

from si.data.dataset import Dataset
from si.io.csv_file import read_csv
from si.model_selection.split import stratified_train_test_split, train_test_split
from si.models.knn_classifier import KNNClassifier
from si.models.knn_regressor import KNNRegressor
from si.models.linear_regression import RidgeRegression
from si.models.logistic_regression import LogisticRegression
from si.models.logistic_regression_l1 import LogisticRegressionL1
from si.models.logistic_regression_momentum import LogisticRegressionMomentum
from si.models.ridge_regression_least_squares import RidgeRegressionLeastSquares
from si.statistics.manhattan_distance import manhattan_distance
from si.statistics.minkowski_distance import minkowski_distance


class TestKNNClassifier(TestCase):

    def setUp(self):
        self.dataset = read_csv(os.path.join(DATASETS_PATH, 'iris', 'iris.csv'), features = True, label = True)
        self.train, self.test = stratified_train_test_split(self.dataset, test_size = 0.2, random_state = 42)

    def test_fit(self):
        knn = KNNClassifier(k = 3)
        self.assertIs(knn.fit(self.train), knn)
        self.assertIs(knn.dataset, self.train)

    def test_predict(self):
        knn = KNNClassifier(k = 3).fit(self.train)
        predictions = knn.predict(self.test)

        self.assertEqual(predictions.shape, (self.test.shape()[0],))
        # the classes are not truncated (strings of different lengths)
        self.assertTrue(set(predictions) <= set(self.dataset.get_classes()))

    def test_k_1_on_training_data(self):
        # with k = 1, each training sample is its own nearest neighbour
        knn = KNNClassifier(k = 1).fit(self.train)
        self.assertEqual(knn.score(self.train), 1.0)

    def test_score(self):
        knn = KNNClassifier(k = 3).fit(self.train)
        self.assertGreater(knn.score(self.test), 0.9)

    def test_other_distances(self):
        for distance in [manhattan_distance, partial(minkowski_distance, p = 3)]:
            knn = KNNClassifier(k = 3, distance = distance).fit(self.train)
            self.assertGreater(knn.score(self.test), 0.9)


class TestKNNRegressor(TestCase):

    def setUp(self):
        self.dataset = read_csv(os.path.join(DATASETS_PATH, 'cpu', 'cpu.csv'), features = True, label = True)
        self.train, self.test = train_test_split(self.dataset, test_size = 0.2, random_state = 42)

    def test_predict(self):
        knn = KNNRegressor(k = 3).fit(self.train)
        predictions = knn.predict(self.test)
        self.assertEqual(predictions.shape, (self.test.shape()[0],))

    def test_average_of_neighbours(self):
        X = np.array([[0.0], [1.0], [2.0], [10.0]])
        y = np.array([1.0, 2.0, 3.0, 100.0])
        knn = KNNRegressor(k = 3).fit(Dataset(X, y))

        prediction = knn.predict(Dataset(np.array([[1.0]])))
        self.assertAlmostEqual(prediction[0], 2.0)

    def test_score(self):
        knn = KNNRegressor(k = 3).fit(self.train)
        score = knn.score(self.test)

        self.assertGreater(score, 0)
        # better than always predicting the mean of the training set
        baseline = np.sqrt(np.mean((self.test.y - self.train.y.mean()) ** 2))
        self.assertLess(score, baseline)


class TestRidgeRegression(TestCase):

    def setUp(self):
        X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]], dtype = float)
        y = np.dot(X, np.array([1, 2])) + 3
        self.dataset = Dataset(X, y)

    def test_gradient_descent(self):
        model = RidgeRegression(l2_penalty = 0, alpha = 0.1, max_iter = 2000).fit(self.dataset)

        prediction = model.predict(Dataset(np.array([[3.0, 5.0]])))
        self.assertAlmostEqual(prediction[0], 16.0, places = 3)
        self.assertLess(model.score(self.dataset), 1e-6)

    def test_cost_history(self):
        model = RidgeRegression(l2_penalty = 1, alpha = 0.01, max_iter = 100).fit(self.dataset)
        costs = list(model.cost_history.values())

        self.assertEqual(len(costs), 100)
        self.assertLess(costs[-1], costs[0])

    def test_patience(self):
        # a learning rate that is too large makes the cost increase -> early stopping
        model = RidgeRegression(alpha = 10, max_iter = 1000, patience = 5).fit(self.dataset)
        self.assertLess(len(model.cost_history), 1000)

    def test_cpu(self):
        dataset = read_csv(os.path.join(DATASETS_PATH, 'cpu', 'cpu.csv'), features = True, label = True)
        train, test = train_test_split(dataset, test_size = 0.2, random_state = 42)
        model = RidgeRegression(l2_penalty = 1, alpha = 0.001, max_iter = 1000).fit(train)

        self.assertEqual(model.theta.shape, (6,))
        self.assertGreater(model.score(test), 0)


class TestRidgeRegressionLeastSquares(TestCase):

    def test_exact_solution(self):
        X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]], dtype = float)
        y = np.dot(X, np.array([1, 2])) + 3
        model = RidgeRegressionLeastSquares(l2_penalty = 0).fit(Dataset(X, y))

        prediction = model.predict(Dataset(np.array([[3.0, 5.0]])))
        self.assertAlmostEqual(prediction[0], 16.0)
        self.assertAlmostEqual(model.score(Dataset(X, y)), 0.0)

    def test_same_as_gradient_descent(self):
        dataset = read_csv(os.path.join(DATASETS_PATH, 'cpu', 'cpu.csv'), features = True, label = True)

        ls = RidgeRegressionLeastSquares(l2_penalty = 1).fit(dataset)
        gd = RidgeRegression(l2_penalty = 1, alpha = 0.01, max_iter = 100000, patience = 100).fit(dataset)

        np.testing.assert_allclose(ls.theta, gd.theta, rtol = 1e-3)
        self.assertAlmostEqual(ls.theta_zero, gd.theta_zero, places = 3)

    def test_penalty_shrinks_coefficients(self):
        dataset = read_csv(os.path.join(DATASETS_PATH, 'cpu', 'cpu.csv'), features = True, label = True)

        small = RidgeRegressionLeastSquares(l2_penalty = 0.1).fit(dataset)
        large = RidgeRegressionLeastSquares(l2_penalty = 1000).fit(dataset)
        self.assertLess(np.sum(large.theta ** 2), np.sum(small.theta ** 2))


class TestLogisticRegression(TestCase):

    def setUp(self):
        self.dataset = read_csv(os.path.join(DATASETS_PATH, 'breast_bin', 'breast-bin.csv'), features = False, label = True)
        self.train, self.test = stratified_train_test_split(self.dataset, test_size = 0.2, random_state = 42)

    def test_predict(self):
        model = LogisticRegression(alpha = 0.01).fit(self.train)
        predictions = model.predict(self.test)

        self.assertEqual(predictions.shape, (self.test.shape()[0],))
        self.assertTrue(set(np.unique(predictions)) <= {0, 1})

    def test_score(self):
        model = LogisticRegression(alpha = 0.01).fit(self.train)
        self.assertGreater(model.score(self.test), 0.9)

    def test_cost_decreases(self):
        model = LogisticRegression(alpha = 0.01, max_iter = 200).fit(self.train)
        costs = list(model.cost_history.values())
        self.assertLess(costs[-1], costs[0])

    def test_momentum(self):
        standard = LogisticRegression(alpha = 0.001, max_iter = 100).fit(self.train)
        momentum = LogisticRegressionMomentum(alpha = 0.001, max_iter = 100, momentum = 0.9).fit(self.train)

        # momentum converges faster: lower training cost after the same number of iterations
        self.assertLess(momentum.cost_history[99], standard.cost_history[99])
        self.assertGreater(momentum.score(self.test), 0.9)

    def test_momentum_zero_is_standard(self):
        standard = LogisticRegression(alpha = 0.01, max_iter = 100).fit(self.train)
        momentum = LogisticRegressionMomentum(alpha = 0.01, max_iter = 100, momentum = 0).fit(self.train)
        np.testing.assert_allclose(momentum.theta, standard.theta)

    def test_l1(self):
        weak   = LogisticRegressionL1(l1_penalty = 10,  alpha = 0.001).fit(self.train)
        strong = LogisticRegressionL1(l1_penalty = 100, alpha = 0.001).fit(self.train)

        # a stronger L1 penalty pushes more coefficients towards zero
        self.assertGreaterEqual(np.sum(np.abs(strong.theta) < 1e-2), np.sum(np.abs(weak.theta) < 1e-2))
        self.assertLess(np.sum(np.abs(strong.theta)), np.sum(np.abs(weak.theta)))
        self.assertGreater(strong.score(self.test), 0.9)
