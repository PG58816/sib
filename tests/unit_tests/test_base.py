from unittest import TestCase

import numpy as np

from si.base.estimator import Estimator
from si.base.model import Model
from si.base.transformer import Transformer
from si.data.dataset import Dataset


class DummyTransformer(Transformer):

    def _fit(self, dataset):
        self.mean = dataset.X.mean(axis = 0)
        return self

    def _transform(self, dataset):
        return Dataset(dataset.X - self.mean, dataset.y)


class DummyModel(Model):

    def _fit(self, dataset):
        self.value = dataset.y.mean()
        return self

    def _predict(self, dataset):
        return np.full(dataset.shape()[0], self.value)

    def _score(self, dataset, predictions):
        return float(np.mean(predictions))


class TestBaseClasses(TestCase):

    def setUp(self):
        self.dataset = Dataset(np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([1.0, 3.0]))

    def test_abstract_classes(self):
        with self.assertRaises(TypeError):
            Estimator()
        with self.assertRaises(TypeError):
            Transformer()
        with self.assertRaises(TypeError):
            Model()

    def test_transformer(self):
        transformer = DummyTransformer()
        self.assertFalse(transformer.is_fitted())

        with self.assertRaises(ValueError):
            transformer.transform(self.dataset)

        self.assertIs(transformer.fit(self.dataset), transformer)
        self.assertTrue(transformer.is_fitted())

        new_dataset = transformer.fit_transform(self.dataset)
        np.testing.assert_allclose(new_dataset.X, [[-1.0, -1.0], [1.0, 1.0]])

    def test_model(self):
        model = DummyModel()
        self.assertFalse(model.is_fitted())

        with self.assertRaises(ValueError):
            model.predict(self.dataset)
        with self.assertRaises(ValueError):
            model.score(self.dataset)

        model.fit(self.dataset)
        self.assertTrue(model.is_fitted())
        np.testing.assert_allclose(model.predict(self.dataset), [2.0, 2.0])
        self.assertAlmostEqual(model.score(self.dataset), 2.0)
        np.testing.assert_allclose(DummyModel().fit_predict(self.dataset), [2.0, 2.0])
