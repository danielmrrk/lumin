from abc import ABC, abstractmethod

import numpy as np

from src.nn.backprop.backprop import Backprop
from src.nn.layer.output_layer import OutputLayer


class Module(ABC):
    def __init__(self, layers: np.array, output_layer: OutputLayer):
        self.layers = layers
        self.output_layer = output_layer

    def fit(self, epochs: int, batch_count: int, X: np.array, y: np.array):
        for _ in range(epochs):

            for batch in range(batches):
                Backprop(self, batch_X, batch_y).backprop()
