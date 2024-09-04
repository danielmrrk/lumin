import numpy as np

from src.nn.activation.activation import Activation
from src.nn.layer.linear import Linear
from src.nn.optimizer.optimizer import Optimizer


class Layer:
    def __init__(self, linear: Linear, activation: Activation):
        self.linear = linear
        self.activation = activation

    def forward_train(self, X: np.array) -> np.array:
        z = self.linear.forward_train(X)
        return self.activation.forward_train(z)

    def forward(self, X: np.array) -> np.array:
        z = self.linear.forward(X)
        return self.activation.forward(z)

    def backprop(self, grad_output: np.array, optimizer: Optimizer) -> np.array:
        grad_output = self.activation.backward_input(grad_output)
        self.linear.backward_params(grad_output)
        grad_output = self.linear.backward_input(grad_output)
        optimizer.update(self.linear.p, self.linear._grad_params)
        return grad_output