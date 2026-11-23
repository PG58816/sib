from unittest import TestCase

import numpy as np

from si.data.dataset import Dataset
from si.statistics.euclidean_distance import euclidean_distance
from si.statistics.f_classification import f_classification
from si.statistics.manhattan_distance import manhattan_distance
from si.statistics.minkowski_distance import minkowski_distance
from si.statistics.sigmoid_function import sigmoid_function


class TestDistances(TestCase):

    def setUp(self):
        self.x = np.array([1.0, 2.0, 3.0])
        self.y = np.array([[1.0, 2.0, 3.0],
                           [4.0, 6.0, 3.0],
                           [0.0, 0.0, 0.0]])

    def test_euclidean_distance(self):
        distances = euclidean_distance(self.x, self.y)

        self.assertEqual(distances.shape, (3,))
        np.testing.assert_allclose(distances, [0.0, 5.0, np.sqrt(14)])

    def test_manhattan_distance(self):
        distances = manhattan_distance(self.x, self.y)

        self.assertEqual(distances.shape, (3,))
        np.testing.assert_allclose(distances, [0.0, 7.0, 6.0])

    def test_minkowski_distance(self):
        # p = 1 is the Manhattan distance and p = 2 is the Euclidean distance
        np.testing.assert_allclose(minkowski_distance(self.x, self.y, p = 1), manhattan_distance(self.x, self.y))
        np.testing.assert_allclose(minkowski_distance(self.x, self.y, p = 2), euclidean_distance(self.x, self.y))

        # p = 3: (3^3 + 4^3)^(1/3)
        np.testing.assert_allclose(minkowski_distance(self.x, self.y, p = 3)[1], (27 + 64) ** (1 / 3))


class TestFClassification(TestCase):

    def test_f_classification(self):
        X = np.array([[1.0, 5.0],
                      [1.1, 4.0],
                      [0.9, 6.0],
                      [5.0, 5.5],
                      [5.1, 4.5],
                      [4.9, 5.0]])
        y = np.array([0, 0, 0, 1, 1, 1])

        F, p = f_classification(Dataset(X, y))

        self.assertEqual(F.shape, (2,))
        self.assertEqual(p.shape, (2,))

        # feature 0 separates the classes, feature 1 does not
        self.assertGreater(F[0], F[1])
        self.assertLess(p[0], 0.05)
        self.assertGreater(p[1], 0.05)


class TestSigmoidFunction(TestCase):

    def test_sigmoid_function(self):
        values = sigmoid_function(np.array([-100.0, -2.0, 0.0, 2.0, 100.0]))

        self.assertAlmostEqual(values[2], 0.5)
        self.assertAlmostEqual(values[1] + values[3], 1.0)
        self.assertTrue(np.all((values >= 0) & (values <= 1)))
        self.assertAlmostEqual(values[0], 0.0)
        self.assertAlmostEqual(values[4], 1.0)
