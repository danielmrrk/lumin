from enum import Enum, auto


class ActivationType(Enum):
    RELU = auto()
    SOFTMAX = auto()

class OptimizerType(Enum):
    GRADIENT_DESCENT = auto()
    SGD = auto()

class LossType(Enum):
    CROSS_ENTROPY = auto()
    MSE = auto()

class NormalizationType(Enum):
    MAX = auto()