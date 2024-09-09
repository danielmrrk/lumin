from abc import ABC, abstractmethod
import numpy as np


class Loss(ABC):
    """
    An abstract base class for loss functions used in neural networks.

    This class defines the interface for loss functions that compute the error between
    predicted values and true values and also compute the gradient of the loss with respect
    to the predicted values. Any specific loss function (e.g., Mean Squared Error, Cross-Entropy Loss)
    should inherit from this class and implement the abstract methods.
    """

    @abstractmethod
    def loss(self, y_hat: np.array, y: np.array) -> float:
        """
        Compute the loss between the predicted values and the true values.

        This method calculates the scalar loss value, which represents the error between
        the model's predictions (`y_hat`) and the true labels (`y`). This value is used to
        evaluate the model's performance.

        Args:
            y_hat (np.array): The predicted values from the model, of shape (batch_size,).
            y (np.array): The true labels, of shape (batch_size,).

        Returns:
            float: The computed loss value.
        """
        pass

    @abstractmethod
    def gradient(self, y_hat: np.array, y: np.array) -> np.array:
        """
        Compute the gradient of the loss with respect to the predicted values.

        This method calculates the gradient of the loss function with respect to the predicted
        values (`y_hat`). This gradient is used during backpropagation to update the model's parameters.

        Args:
            y_hat (np.array): The predicted values from the model, of shape (batch_size,).
            y (np.array): The true labels, of shape (batch_size,).

        Returns:
            np.array: The gradient of the loss with respect to the predicted values, of shape (batch_size,).
        """
        pass
