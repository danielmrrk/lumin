import numpy as np

from src.nn.loss.loss import Loss


class CrossEntropy(Loss):
    def loss(self, y_hat: np.array, y: np.array) -> float:
        epsilon = 1e-15
        y_hat = np.clip(y_hat, epsilon, 1 - epsilon)
        correct_class_probs = y_hat[np.arange(len(y)), y]
        return - np.mean(np.log(correct_class_probs))

    def gradient(self, y_hat: np.array, y: np.array) -> np.array:
        # Return a dummy gradient (all ones) to allow the backpropagation to continue
        # since ∂L/∂a is already handled directly in the softmax backward pass.
        # The gradient will not even be used
        return np.array([])
