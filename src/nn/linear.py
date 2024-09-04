import numpy as np

from src.nn.initialization import InitParams
from src.nn.backprop.parameter_gradients import ParameterGradients
from src.utility.parameter import Parameter
from src.utility.type import InitType


class Linear(ParameterGradients, InitParams):
    def __init__(self, input_dim: int, units: int, init: InitType = InitType.HE):
        InitParams.__init__(self, input_dim, units, init)
        self._input = None
        self._grad_params = None

    def forward_train(self, X: np.array):
        self._input = X
        return self.forward(self, X)

    def forward(self, X: np.array) -> np.array:
        self._input = X
        return np.dot(X, self.p[Parameter.COEFFICIENTS]) + self.p[Parameter.INTERCEPTS]

    def backward_params(self, grad_output: np.array):
        if self._input is None:
            raise Exception("Forward pass needs to be done before backward propagation.")

        grad_coeff = np.dot(self._input.T, grad_output)
        grad_intercept = np.sum(grad_output, axis=0)
        self._grad_params = {
            Parameter.COEFFICIENTS: grad_coeff,
            Parameter.INTERCEPTS: grad_intercept
        }

    def backward_input(self, grad_output: np.array) -> np.array:
        return np.dot(grad_output, self.p[Parameter.COEFFICIENTS].T)

    def update_weights(self, alpha: float = 1e-3):
        if self._grad_params is None:
            raise Exception("You can't update the parameter since you have not calculated the gradient yet.")
        self.p[Parameter.COEFFICIENTS] -= alpha * self._grad_params[Parameter.COEFFICIENTS]
        self.p[Parameter.INTERCEPTS] -= alpha * self._grad_params[Parameter.INTERCEPTS]




