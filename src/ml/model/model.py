from abc import ABC, abstractmethod
import numpy as np

from src.ml.loss.loss import Loss
from src.ml.optimizer.optimizer import Optimizer
from src.utility.type import ModelType
from src.utility.parameter import Parameter, Parameters


class Model(ABC):
    def __init__(self, optimizer: Optimizer, loss: Loss, alpha: float, iter: int):
        self.parameters = None
        self.iter = iter
        self.optimizer = optimizer
        self.loss = loss
        self.technical_name: ModelType = self.get_technical_name()
        self.alpha = alpha
        self.iter = iter

    @abstractmethod
    def get_technical_name(self) -> ModelType:
        pass

    def fit(self, X: np.array, y: np.array):
        coeff_count = X.shape[1]
        self.parameters: Parameters = {Parameter.INTERCEPTS: np.random.random((1,)),
                                       Parameter.COEFFICIENTS: np.random.random((coeff_count,))}

        gradient_function = self.optimizer.get_gradient_function(self.technical_name, self.loss.technical_name)
        self.parameters = self.optimizer.fit(parameters=self.parameters, calculate_gradient=gradient_function,
                                             alpha=self.alpha, X=X, y=y, iter=self.iter)

    @abstractmethod
    def predict(self, input: np.array):
        pass
