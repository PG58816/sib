from unittest import TestCase

import numpy as np

from si.data.dataset import Dataset
from si.metrics.nn_accuracy import nn_accuracy
from si.neural_networks.activation import (ActivationLayer, ReLUActivation, SigmoidActivation,
                                           SoftmaxActivation, TanhActivation)
from si.neural_networks.layers import DenseLayer, Dropout, Layer
from si.neural_networks.losses import BinaryCrossEntropy, CategoricalCrossEntropy, MeanSquaredError
from si.neural_networks.neural_network import NeuralNetwork
from si.neural_networks.optimizers import SGD, Adam


class NoUpdate:
    """'Fake' optimizer that does not change the weights."""
    def update(self, w, grad):
        return w


def numerical_gradient(f, x, eps = 1e-6):
    """Numerical gradient of the scalar function f with respect to the array x (finite differences)."""
    grad = np.zeros_like(x)
    for idx in np.ndindex(x.shape):
        original = x[idx]
        x[idx] = original + eps
        f_plus = f()
        x[idx] = original - eps
        f_minus = f()
        x[idx] = original
        grad[idx] = (f_plus - f_minus) / (2 * eps)
    return grad


class TestDenseLayer(TestCase):

    def test_abstract_layer(self):
        with self.assertRaises(TypeError):
            Layer()

    def test_initialize(self):
        np.random.seed(0)
        layer = DenseLayer(4, (3,)).initialize()

        self.assertEqual(layer.weights.shape, (3, 4))
        self.assertEqual(layer.biases.shape, (1, 4))
        self.assertTrue(np.all((layer.weights >= -0.5) & (layer.weights < 0.5)))
        self.assertTrue(np.all(layer.biases == 0))

    def test_forward_propagation(self):
        layer = DenseLayer(4, (3,)).initialize()
        out   = layer.forward_propagation(np.random.rand(10, 3))

        self.assertEqual(out.shape, (10, 4))
        self.assertEqual(layer.parameters(), 3 * 4 + 4)
        self.assertEqual(layer.output_shape(), (4,))
        self.assertEqual(layer.layer_name(), 'DenseLayer')

    def test_backward_propagation_gradients(self):
        np.random.seed(0)
        X, Y = np.random.rand(4, 3), np.random.rand(4, 2)
        loss = MeanSquaredError()

        class Recorder:
            def update(self, w, grad):
                self.grad = grad
                return w

        layer = DenseLayer(2, (3,)).initialize(Recorder())
        out   = layer.forward_propagation(X, training = True)
        input_error = layer.backward_propagation(loss.derivative(Y, out))

        # dE/dW and dE/dB against the numerical gradients
        num_w = numerical_gradient(lambda: loss.loss(Y, layer.forward_propagation(X, True)), layer.weights)
        num_b = numerical_gradient(lambda: loss.loss(Y, layer.forward_propagation(X, True)), layer.biases)
        np.testing.assert_allclose(layer.w_opt.grad, num_w, atol = 1e-8)
        np.testing.assert_allclose(layer.b_opt.grad, num_b, atol = 1e-8)

        # dE/dX against the numerical gradient
        num_x = numerical_gradient(lambda: loss.loss(Y, layer.forward_propagation(X, True)), X)
        np.testing.assert_allclose(input_error, num_x, atol = 1e-8)

    def test_backward_uses_old_weights(self):
        layer = DenseLayer(2, (3,)).initialize(SGD(learning_rate = 1.0))
        X = np.random.rand(4, 3)
        layer.forward_propagation(X, True)
        old_weights = layer.weights.copy()
        error = np.ones((4, 2))

        input_error = layer.backward_propagation(error)
        np.testing.assert_allclose(input_error, error @ old_weights.T)
        self.assertFalse(np.allclose(layer.weights, old_weights))


class TestDropout(TestCase):

    def setUp(self):
        np.random.seed(0)
        self.X = np.random.rand(1000, 10)

    def test_training(self):
        dropout = Dropout(probability = 0.5)
        out = dropout.forward_propagation(self.X, training = True)
        kept = out != 0

        self.assertAlmostEqual(np.mean(~kept), 0.5, delta = 0.05)
        np.testing.assert_allclose(out[kept], self.X[kept] * 2)
        np.testing.assert_array_equal(dropout.mask.shape, self.X.shape)

    def test_inference(self):
        dropout = Dropout(probability = 0.5)
        np.testing.assert_array_equal(dropout.forward_propagation(self.X, training = False), self.X)

    def test_backward(self):
        dropout = Dropout(probability = 0.3)
        dropout.forward_propagation(self.X, training = True)
        error = dropout.backward_propagation(np.ones_like(self.X))
        np.testing.assert_array_equal(error, dropout.mask)

    def test_shape_and_parameters(self):
        dropout = Dropout(probability = 0.5)
        dropout.set_input_shape((10,))
        self.assertEqual(dropout.output_shape(), (10,))
        self.assertEqual(dropout.parameters(), 0)


class TestActivations(TestCase):

    def setUp(self):
        self.x = np.array([-2.5, -1.2, -0.3, 0.4, 1.1, 2.7])

    def test_abstract_activation(self):
        with self.assertRaises(TypeError):
            ActivationLayer()

    def test_derivatives(self):
        eps = 1e-6
        for act in [SigmoidActivation(), ReLUActivation(), TanhActivation()]:
            num = (act.activation_function(self.x + eps) - act.activation_function(self.x - eps)) / (2 * eps)
            np.testing.assert_allclose(act.derivative(self.x), num, atol = 1e-6)

    def test_relu(self):
        relu = ReLUActivation()
        np.testing.assert_array_equal(relu.activation_function(np.array([-1.0, 0.0, 2.0])), [0.0, 0.0, 2.0])

    def test_ranges(self):
        big = np.array([-100.0, 0.0, 100.0])
        np.testing.assert_allclose(SigmoidActivation().activation_function(big), [0.0, 0.5, 1.0], atol = 1e-12)
        np.testing.assert_allclose(TanhActivation().activation_function(big), [-1.0, 0.0, 1.0], atol = 1e-12)

    def test_softmax(self):
        softmax = SoftmaxActivation()
        X = np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])
        P = softmax.activation_function(X)

        np.testing.assert_allclose(P.sum(axis = 1), [1.0, 1.0])
        np.testing.assert_allclose(P[1], [1 / 3, 1 / 3, 1 / 3])

        # stable version: very large values do not overflow
        P_big = softmax.activation_function(np.array([[1000.0, 1001.0, 1002.0]]))
        self.assertFalse(np.isnan(P_big).any())
        np.testing.assert_allclose(P_big[0], P[0])

    def test_layer(self):
        relu = ReLUActivation()
        relu.set_input_shape((3,))
        out = relu.forward_propagation(np.array([[-1.0, 2.0, 0.5]]), training = True)

        np.testing.assert_array_equal(out, [[0.0, 2.0, 0.5]])
        np.testing.assert_array_equal(relu.backward_propagation(np.ones((1, 3))), [[0.0, 1.0, 1.0]])
        self.assertEqual(relu.output_shape(), (3,))
        self.assertEqual(relu.parameters(), 0)


class TestLosses(TestCase):

    def setUp(self):
        np.random.seed(0)
        self.y_true = np.random.randint(0, 2, size = (5, 1)).astype(float)
        self.y_pred = np.random.uniform(0.05, 0.95, size = (5, 1))

    def check_derivative(self, loss, y_true, y_pred):
        y_pred = y_pred.copy()
        num = numerical_gradient(lambda: loss.loss(y_true, y_pred), y_pred)
        np.testing.assert_allclose(loss.derivative(y_true, y_pred), num, atol = 1e-6)

    def test_mean_squared_error(self):
        mse = MeanSquaredError()
        self.assertAlmostEqual(mse.loss(np.array([1.0, 2.0]), np.array([1.0, 4.0])), 2.0)
        self.check_derivative(mse, self.y_true, self.y_pred)

    def test_binary_cross_entropy(self):
        bce = BinaryCrossEntropy()
        self.assertAlmostEqual(bce.loss(np.array([1.0, 0.0]), np.array([0.8, 0.2])), -2 * np.log(0.8))
        self.check_derivative(bce, self.y_true, self.y_pred)

        # no log(0)
        self.assertTrue(np.isfinite(bce.loss(np.array([1.0]), np.array([0.0]))))

    def test_categorical_cross_entropy(self):
        cce = CategoricalCrossEntropy()
        y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype = float)
        y_pred = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.3, 0.3, 0.4]])

        self.assertAlmostEqual(cce.loss(y_true, y_pred), -(np.log(0.7) + np.log(0.8) + np.log(0.4)))
        self.check_derivative(cce, y_true, y_pred)

        # no log(0) or division by 0
        zero = np.array([[0.0, 1.0, 0.0]])
        self.assertTrue(np.isfinite(cce.loss(y_true[:1], zero)))
        self.assertTrue(np.all(np.isfinite(cce.derivative(y_true[:1], zero))))


class TestOptimizers(TestCase):

    def setUp(self):
        self.w = np.array([[1.0, 2.0], [3.0, 4.0]])
        self.g = np.array([[0.5, -1.0], [2.0, 0.0]])

    def test_sgd(self):
        sgd = SGD(learning_rate = 0.1)
        np.testing.assert_allclose(sgd.update(self.w, self.g), self.w - 0.1 * self.g)

    def test_sgd_momentum(self):
        sgd = SGD(learning_rate = 0.1, momentum = 0.9)
        sgd.update(self.w, self.g)
        np.testing.assert_allclose(sgd.retained_gradient, 0.1 * self.g)
        sgd.update(self.w, self.g)
        np.testing.assert_allclose(sgd.retained_gradient, 0.9 * 0.1 * self.g + 0.1 * self.g)

    def test_sgd_quadratic(self):
        # L(w) = (w - 3)^2, alpha = 0.1, w = 0 -> 0.6 -> 1.08 -> 1.464
        sgd = SGD(learning_rate = 0.1)
        w = np.array([0.0])
        for expected in [0.6, 1.08, 1.464]:
            w = sgd.update(w, 2 * (w - 3))
            self.assertAlmostEqual(w[0], expected)

    def test_adam(self):
        adam = Adam(learning_rate = 0.1)
        w = np.array([0.0])

        # first step moves by ~learning_rate in the direction against the gradient
        w = adam.update(w, 2 * (w - 3))
        self.assertAlmostEqual(w[0], 0.1, places = 6)
        self.assertEqual(adam.t, 1)

        for _ in range(199):
            w = adam.update(w, 2 * (w - 3))
        self.assertAlmostEqual(w[0], 3.0, places = 2)


class TestNeuralNetwork(TestCase):

    def setUp(self):
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype = float)
        y = np.array([0, 1, 1, 0], dtype = float)
        self.xor = Dataset(X, y)

    def build_xor(self, optimizer = SGD, learning_rate = 0.5, epochs = 5000):
        np.random.seed(1)
        net = NeuralNetwork(epochs = epochs, batch_size = 4, optimizer = optimizer, learning_rate = learning_rate,
                            loss = MeanSquaredError, metric = nn_accuracy)
        net.add(DenseLayer(4, (2,)))
        net.add(SigmoidActivation())
        net.add(DenseLayer(1))
        net.add(SigmoidActivation())
        return net

    def test_add(self):
        net = self.build_xor()

        self.assertEqual(len(net.layers), 4)
        self.assertEqual(net.layers[1].input_shape(), (4,))
        self.assertEqual(net.layers[2].weights.shape, (4, 1))
        self.assertEqual([layer.output_shape() for layer in net.layers], [(4,), (4,), (1,), (1,)])

    def test_xor_sgd(self):
        net = self.build_xor().fit(self.xor)
        predictions = net.predict(self.xor)

        self.assertEqual(predictions.shape, (4, 1))
        np.testing.assert_allclose(predictions.round(2).ravel(), [0.05, 0.95, 0.95, 0.07])
        self.assertEqual(net.score(self.xor), 1.0)

    def test_xor_adam(self):
        net = self.build_xor(optimizer = Adam, learning_rate = 0.05, epochs = 2000).fit(self.xor)
        self.assertEqual(net.score(self.xor), 1.0)

    def test_history(self):
        net = self.build_xor(epochs = 100).fit(self.xor)

        self.assertEqual(len(net.history), 100)
        self.assertEqual(set(net.history[1].keys()), {'loss', 'metric'})
        self.assertLess(net.history[100]['loss'], net.history[1]['loss'])

    def test_mini_batches(self):
        net = NeuralNetwork(batch_size = 3)
        X = np.arange(20).reshape(10, 2).astype(float)
        y = np.arange(10).reshape(10, 1).astype(float)
        batches = list(net._get_mini_batches(X, y))

        # 10 samples, batch size 3 -> 3 + 3 + 3 + 1; every sample used once
        self.assertEqual([len(b[0]) for b in batches], [3, 3, 3, 1])
        self.assertEqual(sorted(np.concatenate([b[1] for b in batches]).ravel()), list(range(10)))

    def test_binary_classification(self):
        np.random.seed(42)
        X = np.random.randn(200, 8)
        y = (X @ np.random.randn(8) > 0).astype(float)
        dataset = Dataset(X, y)

        np.random.seed(0)
        net = NeuralNetwork(epochs = 50, batch_size = 16, optimizer = SGD, learning_rate = 0.01,
                            loss = BinaryCrossEntropy, metric = nn_accuracy)
        net.add(DenseLayer(4, (8,)))
        net.add(ReLUActivation())
        net.add(DenseLayer(1))
        net.add(SigmoidActivation())
        net.fit(dataset)

        self.assertGreater(net.score(dataset), 0.9)
