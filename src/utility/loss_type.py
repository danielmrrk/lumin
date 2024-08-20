from enum import Enum, auto


class LossType(Enum):
    MSE = auto()
    CROSS_ENTROPY = auto()
