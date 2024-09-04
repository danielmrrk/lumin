from abc import ABC, abstractmethod

import numpy as np


class Loss(ABC):
    @abstractmethod
    def loss(self, y_hat: np.array, y: np.array) -> float:
        pass

    @abstractmethod
    def gradient(self, y_hat: np.array, y: np.array) -> np.array:
        pass
