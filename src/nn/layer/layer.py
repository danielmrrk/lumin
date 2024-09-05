import numpy as np

from src.nn.activation.activation_factory import activation_factory
from src.nn.activation.activation import Activation
from src.nn.layer.linear import Linear
from src.nn.optimizer.optimizer import Optimizer
from src.nn.type import ActivationType


class Layer:
    def __init__(self, linear: Linear, activation_type: ActivationType):
        self.linear = linear
        self.activation: Activation = activation_factory[activation_type]()

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
        optimizer.update(self.linear.p, self.linear.grad_params())
        return grad_output