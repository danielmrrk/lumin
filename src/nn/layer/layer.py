import numpy as np

from src.nn.activation.activation_factory import activation_factory
from src.nn.activation.activation import Activation
from src.nn.layer.linear import Linear
from src.nn.normalization.norm_factory import norm_factory
from src.nn.optimizer.optimizer import Optimizer
from src.nn.type import ActivationType, NormalizationType


class Layer:
    def __init__(self, linear: Linear, activation_type: ActivationType, norm: NormalizationType = None):
        self.linear = linear
        self.activation: Activation = activation_factory[activation_type]()
        self.norm_func = norm_factory[norm] if norm else None


    def forward_train(self, X: np.array) -> np.array:
        z = self.linear.forward_train(X)
        norm_z = self.normalize(z)
        return self.activation.forward_train(norm_z)

    def forward(self, X: np.array) -> np.array:
        z = self.linear.forward(X)
        norm_z = self.normalize(z)
        return self.activation.forward(norm_z)

    def backprop(self, grad_output: np.array, optimizer: Optimizer, kwargs: dict = None) -> np.array:
        grad_output = self.activation.backward_input(grad_output, kwargs)
        self.linear.backward_params(grad_output)
        grad_output = self.linear.backward_input(grad_output)
        optimizer.update(self.linear.p, self.linear.grad_params())
        return grad_output

    def normalize(self, z: np.array) -> np.array:
        if self.norm_func:
            return self.norm_func(z)
        return z