from abc import ABC, abstractmethod
import numpy as np


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
