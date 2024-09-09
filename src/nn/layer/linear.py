import numpy as np

from src.nn.data_preparation.initialization import InitParams
from src.nn.backprop.parameter_gradients import ParameterGradients
from src.nn.normalization.norm_factory import norm_factory
from src.nn.type import NormalizationType
from src.utility.parameter import Parameter
from src.utility.type import InitType


class Linear(ParameterGradients, InitParams):
    def __init__(self, input_dim: int, units: int, init: InitType = InitType.HE):
        InitParams.__init__(self, input_dim, units, init)
        self.__input = None
        self.__grad_params = None

    def forward_train(self, X: np.array) -> np.array:
        self.__input = X
        return self.forward(X)

    def forward(self, X: np.array) -> np.array:
        return np.dot(X, self.p[Parameter.COEFFICIENTS]) + self.p[Parameter.INTERCEPTS]

    def backward_params(self, grad_output: np.array):
        if self.__input is None:
            raise Exception("Forward pass needs to be done before backward propagation.")

        grad_coeff = np.dot(self.__input.T, grad_output)
        grad_intercept = np.sum(grad_output, axis=0)
        self.__grad_params = {
            Parameter.COEFFICIENTS: grad_coeff,
            Parameter.INTERCEPTS: grad_intercept
        }

    def backward_input(self, grad_output: np.array) -> np.array:
        return np.dot(grad_output, self.p[Parameter.COEFFICIENTS].T)

    def grad_params(self) -> dict:
        return self.__grad_params