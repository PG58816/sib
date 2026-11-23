import os
from unittest import TestCase

import numpy as np

from datasets import DATASETS_PATH

from si.data.dataset import Dataset
from si.feature_selection.select_k_best import SelectKBest
from si.feature_selection.variance_threshold import VarianceThreshold
from si.io.csv_file import read_csv


class TestVarianceThreshold(TestCase):

    def setUp(self):
        self.dataset = Dataset(np.array([[1.0, 0.0, 5.0],
                                         [1.0, 2.0, 5.1],
                                         [1.0, 4.0, 4.9]]),
                               np.array([0, 1, 0]),
                               features = ['a', 'b', 'c'], label = 'y')

    def test_fit(self):
        selector = VarianceThreshold(threshold = 0.1)
        self.assertIs(selector.fit(self.dataset), selector)
        np.testing.assert_allclose(selector.variance, np.var(self.dataset.X, axis = 0))

    def test_transform(self):
        # variances: a = 0, b = 2.67, c = 0.0067
        new_dataset = VarianceThreshold(threshold = 0.1).fit_transform(self.dataset)

        self.assertEqual(new_dataset.shape(), (3, 1))
        self.assertEqual(new_dataset.features, ['b'])
        np.testing.assert_array_equal(new_dataset.y, self.dataset.y)

    def test_threshold_zero(self):
        # removes only the constant feature
        new_dataset = VarianceThreshold(threshold = 0.0).fit_transform(self.dataset)
        self.assertEqual(new_dataset.features, ['b', 'c'])


class TestSelectKBest(TestCase):

    def setUp(self):
        self.dataset = read_csv(os.path.join(DATASETS_PATH, 'iris', 'iris.csv'), features = True, label = True)

    def test_fit(self):
        selector = SelectKBest(k = 2)
        self.assertIs(selector.fit(self.dataset), selector)

        self.assertEqual(selector.F.shape, (4,))
        self.assertEqual(selector.p.shape, (4,))

    def test_transform(self):
        new_dataset = SelectKBest(k = 2).fit_transform(self.dataset)

        self.assertEqual(new_dataset.shape(), (150, 2))
        self.assertEqual(new_dataset.features, ['petal_length', 'petal_width'])
        np.testing.assert_array_equal(new_dataset.y, self.dataset.y)

    def test_k_zero(self):
        new_dataset = SelectKBest(k = 0).fit_transform(self.dataset)
        self.assertEqual(new_dataset.shape(), (150, 0))
