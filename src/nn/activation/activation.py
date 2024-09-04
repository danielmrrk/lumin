from abc import abstractmethod, ABC

import numpy as np


class Activation(ABC):
    def __init__(self):
        self._z = None

    @abstractmethod
    def forward_train(self, z: np.array) -> np.array:
        """
        Compute the forward pass given input z and also saves the input z.
        Should only be used for training

        Args:
            z (np.array): Input data.

        Returns:
            np.array: Output h of the layer after forward pass.
        """

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

        self.__z (np.array): Output f of the linear layer, which is the input to the
        activation function.

        Returns:
            np.array: Gradient of the loss with respect to the input x.
        """
        pass