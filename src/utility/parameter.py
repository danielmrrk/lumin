from enum import Enum
from typing import Dict
import numpy as np


class Parameter(Enum):
    INTERCEPTS = "intercepts"
    COEFFICIENTS = "coefficients"


Parameters = Dict[Parameter, np.array]

