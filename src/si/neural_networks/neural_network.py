from typing import Callable

import numpy as np

from si.data.dataset import Dataset
from si.metrics.nn_accuracy import nn_accuracy
from si.neural_networks.layers import Layer
from si.neural_networks.losses import LossFunction, MeanSquaredError
from si.neural_networks.optimizers import Optimizer, SGD


class NeuralNetwork:

    """
    Neural network: connects the layers, the loss function and the optimizer,
    and trains the network with forward and backward propagation over mini-batches.
    """

    def __init__(self, 
                 epochs:        int      = 100, 
                 batch_size:    int      = 128, 
                 optimizer:     type     = SGD, 
                 learning_rate: float    = 0.01, 
                 verbose:       bool     = False, 
                 loss:          type     = MeanSquaredError, 
                 metric:        Callable = nn_accuracy, 
                 **kwargs):

        """
        Initializes the NeuralNetwork.

        Parameters
        ----------
        epochs: int
            The number of epochs to train the neural network
        batch_size: int
            The batch size to use for training the neural network
        optimizer: Optimizer class
            The optimizer to use for training the neural network
        learning_rate: float
            The learning rate to use for training the neural network
        verbose: bool
            Whether to print the loss and the metric at each epoch
        loss: LossFunction class
            The loss function to use for training the neural network
        metric: Callable
            The metric to use for training the neural network
        **kwargs
            Additional keyword arguments passed to the optimizer
        """

        self.epochs     = epochs
        self.batch_size = batch_size
        self.optimizer  = optimizer(learning_rate = learning_rate, **kwargs)
        self.verbose    = verbose
        self.loss       = loss()
        self.metric     = metric

        self.layers  = []
        self.history = {}

    def add(self, 
            layer: Layer) -> 'NeuralNetwork':

        """
        Adds a layer to the neural network.

        Parameters
        ----------
        layer: Layer
            The layer to add to the neural network

        Returns
        -------
        self: NeuralNetwork
            The neural network
        """

        if self.layers:
            layer.set_input_shape(self.layers[-1].output_shape())

        if hasattr(layer, 'initialize'):
            layer.initialize(self.optimizer)

        self.layers.append(layer)
        return self

    def _get_mini_batches(self, 
                          X:       np.ndarray, 
                          y:       np.ndarray, 
                          shuffle: bool = True):

        """
        Splits the data into mini-batches of size self.batch_size.

        Parameters
        ----------
        X: np.ndarray
            The feature matrix
        y: np.ndarray
            The labels
        shuffle: bool
            Whether to shuffle the samples before splitting

        Yields
        ------
        X_batch, y_batch: np.ndarray
            One mini-batch
        """

        n_samples = X.shape[0]
        indices   = np.arange(n_samples)

        if shuffle:
            np.random.shuffle(indices)

        for start in range(0, n_samples, self.batch_size):
            batch_idxs = indices[start:start + self.batch_size]
            yield X[batch_idxs], y[batch_idxs]

    def _forward_propagation(self, 
                             X:        np.ndarray, 
                             training: bool) -> np.ndarray:

        """
        Performs the forward propagation through all layers (1 -> n).

        Parameters
        ----------
        X: np.ndarray
            The input data
        training: bool
            Whether the network is in training mode or not

        Returns
        -------
        output: np.ndarray
            The output of the last layer
        """

        output = X
        for layer in self.layers:
            output = layer.forward_propagation(output, training)
        return output

    def _backward_propagation(self, 
                              error: np.ndarray) -> np.ndarray:

        """
        Performs the backward propagation through all layers (n -> 1).

        Parameters
        ----------
        error: np.ndarray
            The error of the last layer (derivative of the loss)

        Returns
        -------
        error: np.ndarray
            The error of the first layer
        """

        for layer in reversed(self.layers):
            error = layer.backward_propagation(error)
        return error

    @staticmethod
    def _as_column(y: np.ndarray) -> np.ndarray:

        """
        Converts a 1D label vector (n_samples,) into a column (n_samples, 1),
        so that it has the same shape as the output of the network.
        """

        return y.reshape(-1, 1) if y.ndim == 1 else y

    def fit(self, 
            dataset: Dataset) -> 'NeuralNetwork':

        """
        Trains the neural network on the given dataset.

        Parameters
        ----------
        dataset: Dataset
            The dataset to train the neural network on

        Returns
        -------
        self: NeuralNetwork
            The trained neural network
        """

        X = dataset.X
        y = self._as_column(dataset.y)

        self.history = {}

        for epoch in range(1, self.epochs + 1):

            y_pred_epoch = []
            y_true_epoch = []

            for X_batch, y_batch in self._get_mini_batches(X, y):

                output = self._forward_propagation(X_batch, training = True)

                error = self.loss.derivative(y_batch, output)

                self._backward_propagation(error)

                y_pred_epoch.append(output)
                y_true_epoch.append(y_batch)

            y_pred_epoch = np.concatenate(y_pred_epoch)
            y_true_epoch = np.concatenate(y_true_epoch)

            loss = self.loss.loss(y_true_epoch, y_pred_epoch)

            metric = self.metric(y_true_epoch, y_pred_epoch) if self.metric is not None else None

            self.history[epoch] = {'loss': loss, 'metric': metric}

            if self.verbose:
                message = f'Epoch {epoch}/{self.epochs} - loss: {loss:.4f}'
                if metric is not None:
                    message += f' - {self.metric.__name__}: {metric:.4f}'
                print(message)

        return self

    def predict(self, 
                dataset: Dataset) -> np.ndarray:

        """
        Makes predictions on new data (forward propagation with training = False).

        Parameters
        ----------
        dataset: Dataset
            The dataset to predict

        Returns
        -------
        predictions: np.ndarray
            The output of the network
        """

        return self._forward_propagation(dataset.X, training = False)

    def score(self, 
              dataset: Dataset) -> float:

        """
        Evaluates the neural network on the given dataset with self.metric.

        Parameters
        ----------
        dataset: Dataset
            The dataset to evaluate the neural network on

        Returns
        -------
        score: float
            The value of the metric
        """

        predictions = self.predict(dataset)
        return self.metric(self._as_column(dataset.y), predictions)