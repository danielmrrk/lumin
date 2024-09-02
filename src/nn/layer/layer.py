import numpy as np

from src.nn.backprop.backprop_interfaces import Activation
from src.nn.linear import Linear


class Layer:
    def __init__(self, linear: Linear, activation: Activation):
        self.linear = linear
        self.activation = activation

    def forward(self, X: np.array) -> np.array:
        z = self.linear.forward(X)
        return self.activation.forward(z)

    def backprop(self, grad_output: np.array) -> np.array:
        grad_output = self.activation.backward_input(grad_output)
        grad_output = self.linear.backward_input(grad_output)
        self.linear.backward_params(grad_output)
        return grad_output