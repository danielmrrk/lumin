from enum import Enum, auto


class ActivationType(Enum):
    RELU = auto()

class OptimizerType(Enum):
    GRADIENT_DESCENT = auto()
    SGD = auto()

class LossType(Enum):
    MSE = auto()