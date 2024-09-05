import numpy as np

from src.nn.activation.activation import Activation
from src.nn.layer.layer import Layer
from src.nn.layer.linear import Linear
from src.nn.optimizer.optimizer import Optimizer
from src.nn.type import ActivationType


class OutputLayer(Layer):
    def __init__(self, linear: Linear, activation: ActivationType = None):
        if activation is None:
            self.linear = linear
            self.activation = None
        else:
            super().__init__(linear, activation)

    def forward_train(self, X: np.array):
        if self.activation is None:
            return self.linear.forward_train(X)
        return super().forward_train(X)

    def forward(self, X: np.array) -> np.array:
        if self.activation is None:
            return self.linear.forward(X)
        return super().forward(X)

    def backprop(self, grad_output: np.array, optimizer: Optimizer) -> np.array:
        if self.activation is None:
            self.linear.backward_params(grad_output)
            grad_output = self.linear.backward_input(grad_output)
            optimizer.update(self.linear.p, self.linear.grad_params())
            return grad_output
        return super().backprop(grad_output, optimizer)