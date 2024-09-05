import numpy as np

from src.nn.loss.loss import Loss


class MSE(Loss):
    def loss(self, y_hat: np.array, y: np.array) -> float:
        return np.sum((y_hat - y)**2)

    def gradient(self, y_hat: np.array, y: np.array) -> np.array:
        return (y_hat - y) / len(y)
