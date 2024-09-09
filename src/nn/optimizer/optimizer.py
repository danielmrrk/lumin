from abc import ABC, abstractmethod
import numpy as np
from src.utility.parameter import Parameters


class Optimizer(ABC):
    """
    An abstract base class for optimizers used in training neural networks.

    This class defines a common interface for optimizers, which are responsible for updating
    the parameters of a neural network based on the gradients computed during backpropagation.
    Different optimization algorithms (like SGD, Adam, etc.) should inherit from this class and
    implement the `update` method.

    Attributes:
        mini_batch (bool): A flag indicating whether the optimizer uses mini-batch training.
                           If True, the optimizer is designed to work with mini-batch gradient descent.
    """

    def __init__(self, mini_batch: bool = True):
        """
        Initializes the optimizer with a specified mode for mini-batch training.

        Args:
            mini_batch (bool): Whether to use mini-batch training. Default is True.
        """
        self.mini_batch = mini_batch

    @abstractmethod
    def update(self, p: Parameters, gradient: Parameters, id: int):
        """
        Update the model parameters based on the computed gradients.

        This method should be implemented by subclasses to apply the optimization algorithm to
        update the parameters `p` using the gradients provided in `gradient`.

        Args:
            p (Parameters): The parameters of the model to be updated.
            gradient (Parameters): The gradients of the loss function with respect to the parameters.
            id (int): Unique identifier for each linear layer, which can be used to help access layer specific
            information

        Raises:
            NotImplementedError: This is an abstract method and should be implemented in subclasses.
        """
        pass
