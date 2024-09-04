import numpy as np

from src.nn.activation.activation import Activation

class ReLu(Activation):
    def forward_train(self, z: np.array) -> np.array:
        self._z = z
        return self.forward(z)

    def forward(self, z: np.array) -> np.array:
        self._z = z
        return np.maximum(0, z)

    def backward_input(self, grad_output: np.array) -> np.array:
        if self._z is None:
            raise Exception("Forward pass needs to be done before backward propagation.")
        relu_grad = (self._z > 0).astype(float)
        return grad_output * relu_grad
