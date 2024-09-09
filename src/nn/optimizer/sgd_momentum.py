import numpy as np
from src.nn.optimizer.sgd import SGD
from src.utility.parameter import Parameters, Parameter


class SGDMomentum(SGD):
        def __init__(self, lr: float = 1e-2, momentum = 0.9):
            super().__init__(lr=lr)
            self.momentum = momentum
            self.velocity: dict[int, Parameters] = {}

        def update_velocity(self, p: Parameters, gradient: Parameters, id: int) -> Parameters:
            try:
                specific_velocity = self.velocity[id]
            except KeyError:
                self.velocity[id] = {}
                self.velocity[id][Parameter.COEFFICIENTS] = np.zeros_like(p[Parameter.COEFFICIENTS])
                self.velocity[id][Parameter.INTERCEPTS] = np.zeros_like(p[Parameter.INTERCEPTS])
                specific_velocity = self.velocity[id]
            self.velocity[id][Parameter.COEFFICIENTS] = (self.momentum * specific_velocity[Parameter.COEFFICIENTS]
                                                         + (1 - self.momentum) * gradient[Parameter.COEFFICIENTS])
            self.velocity[id][Parameter.INTERCEPTS] = (self.momentum * specific_velocity[Parameter.INTERCEPTS]
                                                       + (1 - self.momentum) * gradient[Parameter.INTERCEPTS])
            return self.velocity[id]

        def update(self, p: Parameters, gradient: Parameters, id: int):
            updated_velocity = self.update_velocity(p, self.velocity[id], id)
            super().update(p, updated_velocity, id)

