from abc import ABC, abstractmethod

import numpy as np

from src.utility.parameter import Parameters


class Optimizer(ABC):
    def __init__(self, mini_batch: bool = True):
        self.mini_batch = mini_batch

    @abstractmethod
    def update(self, p: Parameters, gradient: Parameters):
        """
        Update parameters using gradient descent.

        Args:
            p: The parameters to be updated.
            gradient: The gradient to be applied for the update.
        """
        pass

