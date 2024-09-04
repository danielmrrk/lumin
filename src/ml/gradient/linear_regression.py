import numpy as np

from src.utility.parameter import Parameters, Parameter


def gradient_linear_regression(p: Parameters, X: np.array, y: np.array) -> Parameters:
    examples = X.shape[0]
    y_hat = np.matmul(X, p[Parameter.COEFFICIENTS]) + p[Parameter.INTERCEPTS][0]
    errors = y_hat - y
    coeff_grad = np.matmul(errors, X) / examples
    intercept_gradient = np.sum(errors) / examples

    return {
        Parameter.COEFFICIENTS: coeff_grad,
        Parameter.INTERCEPTS: intercept_gradient
    }


