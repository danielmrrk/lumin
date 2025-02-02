from typing import Dict, Callable

import numpy as np

from .linear_regression import gradient_linear_regression
from src.utility.type import LossType, ModelType
from src.utility.parameter import Parameters

GradientFunction = Callable[[Parameters, np.array, np.array], Parameters]

gradient_mapping: Dict[ModelType, Dict[LossType, GradientFunction]] = {
    ModelType.LINEAR_REGRESSION: {
        LossType.MSE: gradient_linear_regression
    }
}
