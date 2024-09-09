import numpy as np

from src.nn.optimizer.sgd import SGD
from src.utility.parameter import Parameters, Parameter


class RMSProp(SGD):
    def __init__(self, lr: float = 1e-3, decay_rate=0.9):
        super().__init__(lr=lr)
        self.decay_rate = decay_rate
        self.squared_gradients: dict[int, Parameters] = {}

    def update_squared_gradients(self, p: Parameters, gradient: Parameters, id: int):
        try:
            specific_squared_gradients = self.squared_gradients[id]
        except KeyError:
            self.squared_gradients[id] = {}
            self.squared_gradients[id][Parameter.COEFFICIENTS] = np.zeros_like(p[Parameter.COEFFICIENTS])
            self.squared_gradients[id][Parameter.INTERCEPTS] = np.zeros_like(p[Parameter.INTERCEPTS])
            specific_squared_gradients = self.squared_gradients[id]
        self.squared_gradients[id][Parameter.COEFFICIENTS] = (self.decay_rate * specific_squared_gradients[Parameter.COEFFICIENTS]
                                                              + (1 - self.decay_rate) * gradient[Parameter.COEFFICIENTS]**2)
        self.squared_gradients[id][Parameter.INTERCEPTS] = (self.decay_rate * specific_squared_gradients[Parameter.INTERCEPTS]
                                                            + (1 - self.decay_rate) * gradient[Parameter.INTERCEPTS]**2)
        return self.squared_gradients[id]

    def update(self, p: Parameters, gradient: Parameters, id: int):
        epsilon = 1e-8
        updated_squared_gradients = self.update_squared_gradients(p, gradient, id)
        gradient[Parameter.COEFFICIENTS] = (gradient[Parameter.COEFFICIENTS] /
                                            (np.sqrt(updated_squared_gradients[Parameter.COEFFICIENTS]) + epsilon))
        gradient[Parameter.INTERCEPTS] = (gradient[Parameter.INTERCEPTS] /
                                          (np.sqrt(updated_squared_gradients[Parameter.INTERCEPTS]) + epsilon))
        super().update(p, gradient, id)