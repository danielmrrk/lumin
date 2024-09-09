from abc import abstractmethod, ABC
import numpy as np


class Activation(ABC):
    """
    Abstract base class for activation functions in neural networks.

    This class defines the interface that all activation functions must follow.
    Subclasses are required to implement the methods for forward passes during
    training and inference, as well as the backward pass for gradient computation.

    Attributes:
        __z (np.array): Cached input to the activation function during the forward pass,
                        used for backpropagation during the backward pass.
    """

    def __init__(self):
        """
        Initializes the Activation class, setting up the cached input __z to None.
        """
        self.__z = None

    @abstractmethod
    def forward_train(self, z: np.array) -> np.array:
        """
        Perform the forward pass during training and cache the input for backpropagation.

        Subclasses must implement this method to compute the output of the activation
        function using the input `z` and store `z` for backpropagation. This method is
        called during training, so the input must be cached for use in the backward pass.

        Args:
            z (np.array): Input data, typically the output of the previous layer's linear transformation.

        Returns:
            np.array: The output of the activation function applied to the input `z`.
        """
        pass

    @abstractmethod
    def forward(self, z: np.array) -> np.array:
        """
        Perform the forward pass during inference without caching the input.

        Subclasses must implement this method to compute the output of the activation
        function using the input `z` without storing it. This method is used during
        inference, so caching the input is not required.

        Args:
            z (np.array): Input data, typically the output of the previous layer's linear transformation.

        Returns:
            np.array: The output of the activation function applied to the input `z`.
        """
        pass

    @abstractmethod
    def backward_input(self, grad_output: np.array, kwargs: dict = None) -> np.array:
        """
        Compute the gradient of the loss with respect to the input of the activation function.

        Subclasses must implement this method to compute the gradient of the loss
        function with respect to the input `z` of the activation function. This is
        required for backpropagation, where the gradient is passed backward through the
        network.

        Args:
            grad_output (np.array): Gradient of the loss with respect to the output of the activation function.
            kwargs: Some Activation need additional arguments, like softmax needs the y class indices

        Returns:
            np.array: The gradient of the loss with respect to the input `z` of the activation function.
        """
        pass

    def z(self) -> np.array:
        """
        Get the cached input `z` from the last forward pass.

        Subclasses do not need to implement this method. It returns the input `z`
        that was cached during the most recent forward pass, which is used during
        backpropagation to compute gradients with respect to the input.

        Returns:
            np.array: The input to the activation function from the last forward pass.
        """
        return self.__z
