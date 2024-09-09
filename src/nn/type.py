from enum import Enum, auto


class ActivationType(Enum):
    RELU = auto()
    SOFTMAX = auto()

class OptimizerType(Enum):
    ADAM = auto()
    RMS_PROP = auto()
    SGD_MOMENTUM = auto()
    GRADIENT_DESCENT = auto()
    SGD = auto()

class LossType(Enum):
    CROSS_ENTROPY = auto()
    MSE = auto()

class NormalizationType(Enum):
    MAX = auto()