import unittest
from unittest import TestCase

import numpy as np

from si.data.dataset import Dataset

class TestDataset(unittest.TestCase):

    def test_dataset_construction(self):

        X = np.array([[1, 2, 3], [4, 5, 6]])
        y = np.array([1, 2])

        features = np.array(['a', 'b', 'c'])
        label    = 'y'
        dataset  = Dataset(X, y, features, label)

        self.assertEqual(2.5, dataset.get_mean()[0])
        self.assertEqual((2, 3), dataset.shape())
        self.assertTrue(dataset.has_label())
        self.assertEqual(1, dataset.get_classes()[0])
        self.assertEqual(2.25, dataset.get_variance()[0])
        self.assertEqual(1, dataset.get_min()[0])
        self.assertEqual(4, dataset.get_max()[0])
        self.assertEqual(2.5, dataset.summary().iloc[0, 0])

    def test_dataset_from_random(self):
        dataset = Dataset.from_random(10, 5, 3, features=['a', 'b', 'c', 'd', 'e'], label='y')
        self.assertEqual((10, 5), dataset.shape())
        self.assertTrue(dataset.has_label())

class TestDatasetMethods(TestCase):

    def setUp(self):
        self.X = np.array([[1.0, 2.0, 3.0],
                            [4.0, np.nan, 6.0],
                            [np.nan, 8.0, 9.0],
                            [10.0, 11.0, 12.0]])
        self.y = np.array([0, 1, 0, 1])

    def test_dropna(self):
        dataset = Dataset(self.X.copy(), self.y.copy())
        self.assertIs(dataset.dropna(), dataset)

        self.assertEqual(dataset.X.shape[0], 2)
        self.assertEqual(dataset.y.shape[0], 2)
        self.assertFalse(np.isnan(dataset.X).any())
        np.testing.assert_array_equal(dataset.X, np.array([[1.0, 2.0, 3.0],
                                                             [10.0, 11.0, 12.0]]))
        np.testing.assert_array_equal(dataset.y, np.array([0, 1]))

    def test_dropna_no_y(self):
        dataset = Dataset(self.X.copy())
        self.assertIs(dataset.dropna(), dataset)

        self.assertEqual(dataset.X.shape[0], 2)
        self.assertIsNone(dataset.y)

    def test_fillna_with_value(self):
        dataset = Dataset(self.X.copy(), self.y.copy())
        self.assertIs(dataset.fillna(0.0), dataset)

        self.assertFalse(np.isnan(dataset.X).any())
        self.assertEqual(dataset.X[1, 1], 0.0)
        self.assertEqual(dataset.X[2, 0], 0.0)

    def test_fillna_with_mean(self):
        dataset = Dataset(self.X.copy(), self.y.copy())
        self.assertIs(dataset.fillna('mean'), dataset)

        self.assertFalse(np.isnan(dataset.X).any())
        expected_col0_mean = np.nanmean(self.X[:, 0])
        expected_col1_mean = np.nanmean(self.X[:, 1])
        self.assertAlmostEqual(dataset.X[2, 0], expected_col0_mean)
        self.assertAlmostEqual(dataset.X[1, 1], expected_col1_mean)

    def test_fillna_with_median(self):
        dataset = Dataset(self.X.copy(), self.y.copy())
        self.assertIs(dataset.fillna('median'), dataset)

        self.assertFalse(np.isnan(dataset.X).any())
        expected_col0_median = np.nanmedian(self.X[:, 0])
        expected_col1_median = np.nanmedian(self.X[:, 1])
        self.assertAlmostEqual(dataset.X[2, 0], expected_col0_median)
        self.assertAlmostEqual(dataset.X[1, 1], expected_col1_median)

    def test_remove_by_index(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y = np.array([0, 1, 0])
        dataset = Dataset(X.copy(), y.copy())

        self.assertIs(dataset.remove_by_index(1), dataset)

        self.assertEqual(dataset.X.shape[0], 2)
        self.assertEqual(dataset.y.shape[0], 2)
        np.testing.assert_array_equal(dataset.X, np.array([[1.0, 2.0], [5.0, 6.0]]))
        np.testing.assert_array_equal(dataset.y, np.array([0, 0]))

    def test_remove_by_index_no_y(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        dataset = Dataset(X.copy())

        self.assertIs(dataset.remove_by_index(0), dataset)

        self.assertEqual(dataset.X.shape[0], 2)
        self.assertIsNone(dataset.y)

    def test_remove_by_index_out_of_bounds(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        dataset = Dataset(X.copy())

        with self.assertRaises(IndexError):
            dataset.remove_by_index(5)