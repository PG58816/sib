import unittest

import numpy as np

from si.data.dataset import Dataset
from si.feature_selection.select_percentile import SelectPercentile


class TestSelectPercentile(unittest.TestCase):

    def setUp(self):
        self.X = np.array([[1.0, 5.0, 10.0, 3.0],
                           [1.2, 4.0, 10.5, 7.0],
                           [0.9, 6.0,  9.8, 2.0],
                           [1.1, 5.5, 10.2, 8.0],
                           [5.0, 5.2, 20.0, 4.0],
                           [5.2, 4.8, 20.5, 6.0],
                           [4.9, 5.8, 19.8, 3.0],
                           [5.1, 4.5, 20.2, 7.0]])
        self.y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        self.dataset = Dataset(self.X, self.y, features = ['a', 'b', 'c', 'd'], label = 'y')

    def test_fit(self):
        selector = SelectPercentile(percentile = 50)
        self.assertIs(selector.fit(self.dataset), selector)

        self.assertTrue(selector.is_fitted())
        self.assertEqual(selector.F.shape[0], self.dataset.X.shape[1])
        self.assertEqual(selector.p.shape[0], self.dataset.X.shape[1])

    def test_transform(self):
        selector = SelectPercentile(percentile=50)
        selector.fit(self.dataset)
        new_dataset = selector.transform(self.dataset)

        self.assertEqual(new_dataset.X.shape, (8, 2))
        self.assertEqual(new_dataset.features, ['a', 'c'])
        np.testing.assert_array_equal(new_dataset.X, self.X[:, [0, 2]])
        np.testing.assert_array_equal(new_dataset.y, self.y)
        self.assertEqual(new_dataset.label, 'y')

    def test_fit_transform(self):
        selector = SelectPercentile(percentile = 25)
        new_dataset = selector.fit_transform(self.dataset)

        self.assertEqual(new_dataset.X.shape, (8, 1))
        self.assertIn(new_dataset.features[0], ['a', 'c'])

    def test_ties_at_threshold(self):
        F = np.array([1.2, 3.4, 2.1, 5.6, 4.3, 5.6, 7.8, 6.5, 5.6, 3.2])
        p = np.zeros(10)

        X = np.random.rand(5, 10)
        y = np.array([0, 1, 0, 1, 0])
        dataset = Dataset(X, y)

        selector = SelectPercentile(score_func=lambda ds: (F, p), percentile = 40)
        new_dataset = selector.fit_transform(dataset)

        self.assertEqual(new_dataset.X.shape[1], 4)
        self.assertEqual(new_dataset.features, ['feat_3', 'feat_5', 'feat_6', 'feat_7'])

    def test_non_integer_number_of_features(self):
        F = np.arange(10, dtype = float)
        p = np.zeros(10)

        dataset = Dataset(np.random.rand(5, 10), np.array([0, 1, 0, 1, 0]))

        selector = SelectPercentile(score_func = lambda ds: (F, p), percentile = 25)
        new_dataset = selector.fit_transform(dataset)

        self.assertEqual(new_dataset.X.shape[1], 2)
        self.assertEqual(new_dataset.features, ['feat_8', 'feat_9'])

    def test_transform_before_fit(self):
        selector = SelectPercentile(percentile = 50)

        with self.assertRaises(ValueError):
            selector.transform(self.dataset)