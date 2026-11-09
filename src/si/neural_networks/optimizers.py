from abc import abstractmethod

import numpy as np


class Optimizer:

    """
    Base class for the optimizers of a neural network.
    """

    def __init__(self, 
                 learning_rate: float):

        """
        Initializes the Optimizer.

        Parameters
        ----------
        learning_rate: float
            The learning rate of the optimizer
        """

        self.learning_rate = learning_rate

    @abstractmethod
    def update(self, 
               w:           np.ndarray, 
               grad_loss_w: np.ndarray) -> np.ndarray:

        """
        Updates the parameters (weights or biases) of a layer.
        Abstract method that needs to be implemented by all subclasses.

        Parameters
        ----------
        w: np.ndarray
            The current parameters of the layer
        grad_loss_w: np.ndarray
            The gradient of the loss function with respect to the parameters
        """

        raise NotImplementedError


class SGD(Optimizer):

    """
    Stochastic Gradient Descent optimizer, with momentum.
    """

    def __init__(self, 
                 learning_rate: float = 0.01, 
                 momentum:      float = 0.0):

        """
        Initializes the SGD optimizer.

        Parameters
        ----------
        learning_rate: float
            The learning rate to use for updating the weights
        momentum: float
            The momentum to use for updating the weights
        """

        super().__init__(learning_rate)
        self.momentum          = momentum
        self.retained_gradient = None

    def update(self, 
               w:           np.ndarray, 
               grad_loss_w: np.ndarray) -> np.ndarray:

        """
        Updates the retained gradient and computes the updated weights:
        V_t = momentum * V_(t-1) + (1 - momentum) * dL/dW
        W_t = W_(t-1) - learning_rate * V_t

        Parameters
        ----------
        w: np.ndarray
            The current weights
        grad_loss_w: np.ndarray
            The gradient of the loss function with respect to the weights

        Returns
        -------
        w: np.ndarray
            The updated weights
        """

        if self.retained_gradient is None:
            self.retained_gradient = np.zeros(np.shape(w))

        self.retained_gradient = self.momentum * self.retained_gradient + (1 - self.momentum) * grad_loss_w

        return w - self.learning_rate * self.retained_gradient

class Adam(Optimizer):

    """
    Adam optimizer: combines RMSprop (scales the learning rate with the squared gradients)
    and SGD with momentum (uses a moving average of the gradient).
    """

    def __init__(self, 
                 learning_rate: float = 0.001, 
                 beta_1:        float = 0.9, 
                 beta_2:        float = 0.999, 
                 epsilon:       float = 1e-8):

        """
        Initializes the Adam optimizer.

        Parameters
        ----------
        learning_rate: float
            The learning rate to use for updating the weights
        beta_1: float
            The exponential decay rate for the 1st moment estimates
        beta_2: float
            The exponential decay rate for the 2nd moment estimates
        epsilon: float
            A small constant for numerical stability
        """

        super().__init__(learning_rate)
        self.beta_1  = beta_1
        self.beta_2  = beta_2
        self.epsilon = epsilon

        self.m = None
        self.v = None
        self.t = 0

    def update(self, 
               w:           np.ndarray, 
               grad_loss_w: np.ndarray) -> np.ndarray:

        """
        Updates t, m and v and computes the updated weights:
        m_t = beta_1 * m_(t-1) + (1 - beta_1) * dL/dW
        v_t = beta_2 * v_(t-1) + (1 - beta_2) * (dL/dW)^2
        m_hat = m_t / (1 - beta_1^t);  v_hat = v_t / (1 - beta_2^t)
        W_t = W_(t-1) - learning_rate * m_hat / (sqrt(v_hat) + epsilon)

        Parameters
        ----------
        w: np.ndarray
            The current weights
        grad_loss_w: np.ndarray
            The gradient of the loss function with respect to the weights

        Returns
        -------
        w: np.ndarray
            The updated weights
        """

        if self.m is None:
            self.m = np.zeros(np.shape(w))
        if self.v is None:
            self.v = np.zeros(np.shape(w))

        self.t += 1

        self.m = self.beta_1 * self.m + (1 - self.beta_1) * grad_loss_w
        self.v = self.beta_2 * self.v + (1 - self.beta_2) * grad_loss_w ** 2

        m_hat = self.m / (1 - self.beta_1 ** self.t)
        v_hat = self.v / (1 - self.beta_2 ** self.t)

        return w - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)