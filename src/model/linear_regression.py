import numpy as np

from .model import Model
from src.loss.loss import Loss
from src.loss.mse import MSE
from src.optimizer.gradient_descent import GradientDescent
from src.optimizer.optimizer import Optimizer
from src.utility.type import ModelType
from src.utility.parameter import Parameter


class LinearRegression(Model):
    def __init__(self, alpha: float, iter: int, optimizer: Optimizer = GradientDescent(), loss: Loss = MSE()):
        super().__init__(optimizer, loss, alpha, iter)

    def get_technical_name(self) -> ModelType:
        return ModelType.LINEAR_REGRESSION

    def predict(self, input: np.array) -> np.array:
        return np.matmul(input, self.parameters[Parameter.COEFFICIENTS]) + self.parameters[Parameter.INTERCEPTS][0]
