import numpy as np

from src.nn.backprop.backprop_interfaces import Activation
from src.nn.layer.layer import Layer
from src.nn.linear import Linear


class OutputLayer(Layer):
    def __init__(self, linear: Linear, activation: Activation = None):
        super().__init__(linear, activation)

    def forward(self, X: np.array) -> np.array:
        if self.activation is None:
            return self.linear.forward(X)
        return super().forward(X)

    def backprop(self, grad_output: np.array) -> np.array:
        if self.activation is None:
            grad_output = self.linear.backward_input(grad_output)
            self.linear.backward_params(grad_output)
            return grad_output
        return super().backprop(grad_output)