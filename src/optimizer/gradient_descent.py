import numpy as np

from .optimizer import Optimizer
from src.gradient.gradient_mapping import gradient_mapping, GradientFunction
from src.utility.loss_type import LossType
from src.utility.model_type import ModelType
from src.utility.optimizer_type import OptimizerType
from src.utility.parameter import Parameters, Parameter


class GradientDescent(Optimizer):
    def get_technical_name(self) -> OptimizerType:
        return OptimizerType.GRADIENT_DESCENT

    def get_gradient_function(self, model_type: ModelType, loss_type: LossType) -> GradientFunction:
        return gradient_mapping[model_type][loss_type]

    def fit(self, parameters: Parameters, calculate_gradient: GradientFunction, alpha: int, X: np.array,
               y: np.array, iter: int) -> Parameters:
        for _ in range(iter):
            parameter_gradient = calculate_gradient(parameters, X, y)
            parameters[Parameter.COEFFICIENTS] = parameters[Parameter.COEFFICIENTS] - alpha * parameter_gradient[Parameter.COEFFICIENTS]
            parameters[Parameter.INTERCEPTS] = parameters[Parameter.INTERCEPTS] - alpha * parameter_gradient[Parameter.INTERCEPTS]
        return parameters
