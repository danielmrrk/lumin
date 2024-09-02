import numpy as np

from src.utility.parameter import Parameters, Parameter


def round_parameters(p:Parameters, digit:int) -> Parameters:
    p[Parameter.COEFFICIENTS] = np.round(p[Parameter.COEFFICIENTS], digit)
    p[Parameter.INTERCEPTS] = np.round(p[Parameter.INTERCEPTS], digit)
    return p
