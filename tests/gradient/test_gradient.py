import numpy as np
import pytest
from src.utility.parameter import Parameters, Parameter
from src.gradient import linear_regression


def test_gradient_linear_regression():
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([2, 4, 6, 8, 10])

    parameters = {
        Parameter.COEFFICIENTS: np.array([0.5]),
        Parameter.INTERCEPTS: np.array([1.0])
    }

    expected_gradients = {
        Parameter.COEFFICIENTS: np.array([-13.5]),
        Parameter.INTERCEPTS: -3.5
    }

    computed_gradients: Parameters = linear_regression.gradient_linear_regression(parameters, X, y)
    assert np.allclose(computed_gradients[Parameter.COEFFICIENTS], expected_gradients[Parameter.COEFFICIENTS],
                       rtol=1e-5)
    assert np.allclose(computed_gradients[Parameter.INTERCEPTS], expected_gradients[Parameter.INTERCEPTS], rtol=1e-5)

