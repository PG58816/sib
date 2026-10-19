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

        # 1. Weights: uniform random values in [-0.5, 0.5[  (n_features x n_units)
        self.weights = np.random.rand(n_features, self.n_units) - 0.5

        # 2. Biases: zeros  (1 x n_units)
        self.biases = np.zeros((1, self.n_units))

        # 3. One optimizer for the weights and another for the biases
        if optimizer is not None:
            self.w_opt = copy.deepcopy(optimizer)
            self.b_opt = copy.deepcopy(optimizer)

        # 4. Return itself
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

    def output_shape(self) -> tuple:

        """
        Returns the shape of the output of the layer.

        Returns
        -------
        output_shape: tuple
            The shape of the output of the layer (n_units, )
        """

        return (self.n_units, )