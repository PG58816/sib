import copy
from abc import ABCMeta, abstractmethod

import numpy as np


class Layer(metaclass = ABCMeta):

    """
    Abstract base class for the layers of a neural network.
    """

    @abstractmethod
    def forward_propagation(self, 
                            input:    np.ndarray, 
                            training: bool):

        """
        Performs the forward propagation of the layer.
        Abstract method that needs to be implemented by all subclasses.

        Parameters
        ----------
        input: np.ndarray
            The input of the layer
        training: bool
            Whether the layer is in training mode or not
        """

        raise NotImplementedError

    @abstractmethod
    def backward_propagation(self, 
                             output_error: np.ndarray):

        """
        Performs the backward propagation of the layer.
        Abstract method that needs to be implemented by all subclasses.

        Parameters
        ----------
        output_error: np.ndarray
            The output error of the layer (dE/dY)
        """

        raise NotImplementedError

    @abstractmethod
    def output_shape(self):

        """
        Returns the shape of the output of the layer.
        Abstract method that needs to be implemented by all subclasses.
        """

        raise NotImplementedError

    @abstractmethod
    def parameters(self):

        """
        Returns the number of parameters of the layer.
        Abstract method that needs to be implemented by all subclasses.
        """

        raise NotImplementedError

    def set_input_shape(self, 
                        input_shape: tuple):

        """
        Sets the shape of the input of the layer.

        Parameters
        ----------
        input_shape: tuple
            The shape of the input of the layer
        """

        self._input_shape = input_shape

    def input_shape(self):

        """
        Returns the shape of the input of the layer.

        Returns
        -------
        input_shape: tuple
            The shape of the input of the layer
        """

        return self._input_shape

    def layer_name(self):

        """
        Returns the name of the layer (the name of the class).

        Returns
        -------
        layer_name: str
            The name of the layer
        """

        return self.__class__.__name__


class DenseLayer(Layer):

    """
    Dense (fully connected) layer: connects each neuron to every neuron
    in the previous and next layers.
    """

    def __init__(self, 
                 n_units:     int, 
                 input_shape: tuple = None):

        """
        Initializes the DenseLayer.

        Parameters
        ----------
        n_units: int
            The number of neurons in the layer
        input_shape: tuple
            The shape of the input of the layer (for tabular data: (n_features, ))
        """

        super().__init__()
        self.n_units      = n_units
        self._input_shape = input_shape

        self.input   = None
        self.output  = None
        self.weights = None
        self.biases  = None

    def initialize(self, 
                   optimizer = None) -> 'DenseLayer':

        """
        Initializes the layer with random weights and biases and with the optimizer.

        Parameters
        ----------
        optimizer: Optimizer
            The optimizer to use to update the weights and biases of the layer

        Returns
        -------
        self: DenseLayer
            The initialized layer
        """

        n_features = self.input_shape()[0]

        self.weights = np.random.rand(n_features, self.n_units) - 0.5

        self.biases = np.zeros((1, self.n_units))

        if optimizer is not None:
            self.w_opt = copy.deepcopy(optimizer)
            self.b_opt = copy.deepcopy(optimizer)

        return self

    def parameters(self) -> int:

        """
        Returns the number of parameters of the layer (weights + biases).

        Returns
        -------
        parameters: int
            The number of parameters of the layer
        """

        return np.prod(self.weights.shape) + np.prod(self.biases.shape)

    def forward_propagation(self, 
                            input:    np.ndarray, 
                            training: bool = False) -> np.ndarray:

        """
        Performs the forward propagation of the layer: Y = XW + B.

        Parameters
        ----------
        input: np.ndarray
            The input of the layer
        training: bool
            Whether the layer is in training mode or not

        Returns
        -------
        output: np.ndarray
            The output of the layer
        """

        self.input  = input
        self.output = np.dot(self.input, self.weights) + self.biases

        return self.output

    def backward_propagation(self, 
                             output_error: np.ndarray) -> np.ndarray:

        """
        Performs the backward propagation of the layer: computes the input error,
        the weights and biases errors, and updates the weights and biases with the optimizer.

        Parameters
        ----------
        output_error: np.ndarray
            The output error of the layer (dE/dY)

        Returns
        -------
        input_error: np.ndarray
            The input error of the layer (dE/dX), passed on to the previous layer
        """

        input_error = np.dot(output_error, self.weights.T)

        weights_error = np.dot(self.input.T, output_error)

        bias_error = np.sum(output_error, axis = 0, keepdims = True)

        self.weights = self.w_opt.update(self.weights, weights_error)

        self.biases = self.b_opt.update(self.biases, bias_error)

        return input_error

    def output_shape(self) -> tuple:

        """
        Returns the shape of the output of the layer.

        Returns
        -------
        output_shape: tuple
            The shape of the output of the layer (n_units, )
        """

        return (self.n_units, )

class Dropout(Layer):

    """
    Dropout layer: during training, a random set of neurons is temporarily ignored
    (dropped out), which helps prevent overfitting.
    """

    def __init__(self, 
                 probability: float):

        """
        Initializes the Dropout layer.

        Parameters
        ----------
        probability: float
            The dropout rate, between 0 and 1
        """

        super().__init__()
        self.probability = probability

        self.mask   = None
        self.input  = None
        self.output = None

    def forward_propagation(self, 
                            input:    np.ndarray, 
                            training: bool = False) -> np.ndarray:

        """
        Performs the forward propagation of the layer: in training mode, applies a binomial
        mask to the input and scales it; in inference mode, returns the input unchanged.

        Parameters
        ----------
        input: np.ndarray
            The input of the layer
        training: bool
            Whether the layer is in training mode or in inference mode

        Returns
        -------
        output: np.ndarray
            The output of the layer
        """

        self.input = input

        if training:
            scaling_factor = 1 / (1 - self.probability)

            self.mask = np.random.binomial(1, 1 - self.probability, size = input.shape)

            self.output = input * self.mask * scaling_factor
            return self.output

        self.output = input
        return self.output

    def backward_propagation(self, 
                             output_error: np.ndarray) -> np.ndarray:

        """
        Performs the backward propagation of the layer: multiplies the output error by the mask.

        Parameters
        ----------
        output_error: np.ndarray
            The output error of the layer (dE/dY)

        Returns
        -------
        input_error: np.ndarray
            The input error of the layer (dE/dX)
        """

        return output_error * self.mask

    def output_shape(self) -> tuple:

        """
        Returns the shape of the output of the layer (same as the input shape).

        Returns
        -------
        output_shape: tuple
            The shape of the output of the layer
        """

        return self._input_shape

    def parameters(self) -> int:

        """
        Returns the number of parameters of the layer (dropout layers have none).

        Returns
        -------
        parameters: int
            0
        """

        return 0

