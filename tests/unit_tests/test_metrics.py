from unittest import TestCase

import numpy as np

from si.metrics.accuracy import accuracy
from si.metrics.mse import mse
from si.metrics.nn_accuracy import nn_accuracy
from si.metrics.rmse import rmse


class TestAccuracy(TestCase):

    def test_accuracy(self):
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])

        self.assertAlmostEqual(accuracy(y_true, y_pred), 0.8)

    def test_accuracy_strings(self):
        y_true = np.array(['a', 'b', 'c', 'a'])
        y_pred = np.array(['a', 'b', 'a', 'a'])

        self.assertAlmostEqual(accuracy(y_true, y_pred), 0.75)


class TestNNAccuracy(TestCase):

    def test_nn_accuracy_binary(self):
        # column of probabilities -> 0/1 with a 0.5 threshold
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([[0.1], [0.9], [0.4], [0.2]])

        self.assertAlmostEqual(nn_accuracy(y_true, y_pred), 0.75)

    def test_nn_accuracy_multiclass(self):
        # one-hot labels and probabilities -> argmax
        y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        y_pred = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.5, 0.3, 0.2]])

        self.assertAlmostEqual(nn_accuracy(y_true, y_pred), 2 / 3)

    def test_nn_accuracy_labels(self):
        # 1D labels are left unchanged (same as accuracy)
        y_true = np.array([0, 1, 1])
        y_pred = np.array([0, 1, 0])

        self.assertAlmostEqual(nn_accuracy(y_true, y_pred), accuracy(y_true, y_pred))


class TestRegressionMetrics(TestCase):

    def setUp(self):
        self.y_true = np.array([1.0, 2.0, 3.0, 4.0])
        self.y_pred = np.array([1.0, 3.0, 3.0, 2.0])

    def test_mse(self):
        # (0 + 1 + 0 + 4) / 4
        self.assertAlmostEqual(mse(self.y_true, self.y_pred), 1.25)
        self.assertAlmostEqual(mse(self.y_true, self.y_true), 0.0)

    def test_rmse(self):
        self.assertAlmostEqual(rmse(self.y_true, self.y_pred), np.sqrt(1.25))
        self.assertAlmostEqual(rmse(self.y_true, self.y_pred) ** 2, mse(self.y_true, self.y_pred))
