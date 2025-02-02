from abc import ABC, abstractmethod
import numpy as np

from ..gradient.gradient_mapping import GradientFunction
from src.utility.type import LossType, ModelType, OptimizerType
from src.utility.parameter import Parameters


class Optimizer(ABC):
    def __init__(self):
        self.technical_name: OptimizerType = self.get_technical_name()

    @abstractmethod
    def get_technical_name(self) -> OptimizerType:
        pass

    @abstractmethod
    def get_gradient_function(self, model: ModelType, loss: LossType) -> GradientFunction:
        pass

    @abstractmethod
    def fit(self, parameters: Parameters, calculate_gradient: GradientFunction, alpha: float, X: np.array, y: np.array, iter: int) -> Parameters:
        pass
