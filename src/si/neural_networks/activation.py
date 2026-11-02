from abc import abstractmethod

import numpy as np

from si.neural_networks.layers import Layer
from si.statistics.sigmoid_function import sigmoid_function


class ActivationLayer(Layer):

    """
    Base class for the activation layers of a neural network.
    Activation layers introduce non-linearity to the model.
    """

    def __init__(self):

        """
        Initializes the ActivationLayer.
        """

        super().__init__()
        self.input  = None
        self.output = None

    def forward_propagation(self, 
                            input:    np.ndarray, 
                            training: bool = False) -> np.ndarray:

        """
        Performs the forward propagation of the layer by applying the activation function.

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

        self.input  = input
        self.output = self.activation_function(self.input)

        return self.output

    def backward_propagation(self, 
                             output_error: np.ndarray) -> np.ndarray:

        """
        Performs the backward propagation of the layer: applies the derivative of the
        activation function to the layer input and multiplies it (element-wise) by the output error.

        Parameters
        ----------
        output_error: np.ndarray
            The output error of the layer (dE/dY)

        Returns
        -------
        input_error: np.ndarray
            The input error of the layer (dE/dX)
        """

        return self.derivative(self.input) * output_error

    @abstractmethod
    def activation_function(self, 
                            input: np.ndarray) -> np.ndarray:

        """
        Computes the activation function on the given input.
        Abstract method that needs to be implemented by all subclasses.

        Parameters
        ----------
        input: np.ndarray
            The input of the layer
        """

        raise NotImplementedError

    @abstractmethod
    def derivative(self, 
                   input: np.ndarray) -> np.ndarray:

        """
        Computes the derivative of the activation function on the given input.
        Abstract method that needs to be implemented by all subclasses.

        Parameters
        ----------
        input: np.ndarray
            The input of the layer
        """

        raise NotImplementedError

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
        Returns the number of parameters of the layer (activation layers have none).

        Returns
        -------
        parameters: int
            0
        """

        return 0


class ReLUActivation(ActivationLayer):

    """
    ReLU (Rectified Linear Unit) activation layer: outputs the input if it is positive, and zero otherwise.
    """

    def activation_function(self, 
                            input: np.ndarray) -> np.ndarray:

        """
        Applies the ReLU function: f(x) = 0 if x < 0; x if x >= 0.

        Parameters
        ----------
        input: np.ndarray
            The input of the layer

        Returns
        -------
        output: np.ndarray
            The ReLU of the input
        """

        return np.maximum(0, input)

    def derivative(self, 
                   input: np.ndarray) -> np.ndarray:

        """
        Applies the derivative of the ReLU function: f'(x) = 0 if x < 0; 1 if x >= 0.

        Parameters
        ----------
        input: np.ndarray
            The input of the layer

        Returns
        -------
        derivative: np.ndarray
            The derivative of the ReLU on the input
        """

        return np.where(input >= 0, 1, 0)


class SigmoidActivation(ActivationLayer):

    """
    Sigmoid activation layer: compresses the input values to a range between 0 and 1.
    """

    def activation_function(self, 
                            input: np.ndarray) -> np.ndarray:

        """
        Applies the sigmoid function: f(x) = 1 / (1 + e^(-x)).

        Parameters
        ----------
        input: np.ndarray
            The input of the layer

        Returns
        -------
        output: np.ndarray
            The sigmoid of the input
        """

        return sigmoid_function(input)

    def derivative(self, 
                   input: np.ndarray) -> np.ndarray:

        """
        Applies the derivative of the sigmoid function: f'(x) = f(x) * (1 - f(x)).

        Parameters
        ----------
        input: np.ndarray
            The input of the layer

        Returns
        -------
        derivative: np.ndarray
            The derivative of the sigmoid on the input
        """

        f = self.activation_function(input)
        return f * (1 - f)

class TanhActivation(ActivationLayer):

    """
    Tanh activation layer: applies the hyperbolic tangent, squashing the values to the range -1 to 1.
    """

    def activation_function(self, 
                            input: np.ndarray) -> np.ndarray:

        """
        Applies the tanh function: f(x) = (e^x - e^(-x)) / (e^x + e^(-x)).

        Parameters
        ----------
        input: np.ndarray
            The input of the layer

        Returns
        -------
        output: np.ndarray
            The tanh of the input
        """

        return np.tanh(input)

    def derivative(self, 
                   input: np.ndarray) -> np.ndarray:

        """
        Applies the derivative of the tanh function: f'(x) = 1 - f(x)^2.

        Parameters
        ----------
        input: np.ndarray
            The input of the layer

        Returns
        -------
        derivative: np.ndarray
            The derivative of the tanh on the input
        """

        return 1 - self.activation_function(input) ** 2


class SoftmaxActivation(ActivationLayer):

    """
    Softmax activation layer: transforms the raw output scores into a probability
    distribution (that sums to 1), for multi-class classification.
    """

    def activation_function(self, 
                            input: np.ndarray) -> np.ndarray:

        """
        Applies the (stable) softmax function: f(x_i) = e^(x_i) / sum_j e^(x_j).
        The max value is subtracted from x before the exponential to avoid very large values.

        Parameters
        ----------
        input: np.ndarray
            The input of the layer (n_samples, n_classes)

        Returns
        -------
        output: np.ndarray
            The softmax of the input (each row sums to 1)
        """

        exp_values = np.exp(input - np.max(input, axis = -1, keepdims = True))
        return exp_values / np.sum(exp_values, axis = -1, keepdims = True)

    def derivative(self, 
                   input: np.ndarray) -> np.ndarray:

        """
        Applies the derivative of the softmax function: f'(x) = f(x) * (1 - f(x)).

        Parameters
        ----------
        input: np.ndarray
            The input of the layer

        Returns
        -------
        derivative: np.ndarray
            The derivative of the softmax on the input
        """

        f = self.activation_function(input)
        return f * (1 - f)

