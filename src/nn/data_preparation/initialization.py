import numpy as np

from src.utility.parameter import Parameters, Parameter
from src.utility.type import InitType


def he_initialization(input_dim: int, units: int) -> Parameters:
    std = np.sqrt(2 / input_dim)
    return {
        Parameter.COEFFICIENTS: np.random.normal(0, std, (input_dim, units)),
        Parameter.INTERCEPTS: np.random.normal(0, std, (units,))
    }


def xavier_initialization(input_dim: int, units: int) -> Parameters:
    std = np.sqrt(2 / (input_dim + units))
    return {
        Parameter.COEFFICIENTS: np.random.normal(0, std, (input_dim, units)),
        Parameter.INTERCEPTS: np.random.normal(0, std, (units,))
    }


class InitParams:
    def __init__(self, input_dim: int, units: int, init: InitType):
        self.input_dim = input_dim
        self.units = units
        if init == InitType.HE:
            self.p =  he_initialization(input_dim, units)
        else:
            self.p = xavier_initialization(input_dim, units)


