from abc import ABC, abstractmethod
import numpy as np

from src.utility.parameter import Parameters


class Activation(ABC):
    def __init__(self):
        self._z = None

    @abstractmethod
    def forward(self, z: np.array) -> np.array:
        """
        Compute the forward pass given input z.

        Args:
            z (np.array): Input data.

        Returns:
            np.array: Output h of the layer after forward pass.
        """
        pass

    @abstractmethod
    def backward_input(self, grad_output: np.array) -> np.array:
        """
        Compute the gradient of the loss with respect to the input x.

        Args:
            grad_output (np.array): Gradient of the loss with respect to the output.

        self.__z (np.array): Output f of the linear layer, which functions as the input to the
        activation function.

        Returns:
            np.array: Gradient of the loss with respect to the input x.
        """
        pass


class ParameterGradients(ABC):
    @abstractmethod
    def backward_params(self, grad_output: np.array):
        """
        Compute the gradient of the loss with respect to the weights (parameters).

        Args:
            grad_output (np.array): Gradient of the loss with respect to the output.

        self.__input (np.array): Input data from the forward pass. Is None if the forward
        pass wasn't done yet.

        Sets the gradients of the loss with respect to the parameters.
        """
        pass
