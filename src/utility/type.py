from enum import Enum, auto


class LossType(Enum):
    MSE = auto()
    CROSS_ENTROPY = auto()

class ModelType(Enum):
    LINEAR_REGRESSION = auto()
    LOGISTIC_REGRESSION = auto()

class OptimizerType(Enum):

    GRADIENT_DESCENT = auto()

class InitType(Enum):
    HE = auto()
    XAVIER = auto()