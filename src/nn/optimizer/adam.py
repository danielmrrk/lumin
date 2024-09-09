import numpy as np

from src.nn.optimizer.rms_prop import RMSProp
from src.nn.optimizer.sgd import SGD
from src.nn.optimizer.sgd_momentum import SGDMomentum
from src.utility.parameter import Parameters, Parameter


class Adam(SGD):
    def __init__(self, lr: float = 1e-3, momentum: float = 0.9, decay_rate: float = 0.999):
        super().__init__(lr=lr)
        self.rms_prop = RMSProp(decay_rate=decay_rate)
        self.sgd_momentum = SGDMomentum(momentum=momentum)

    def update(self, p: Parameters, gradient: Parameters, id: int):
        epsilon = 1e-8
        updated_velocity = self.sgd_momentum.update_velocity(p, gradient, id)
        updated_squared_gradients = self.rms_prop.update_squared_gradients(p, gradient, id)
        gradient[Parameter.COEFFICIENTS] = (updated_velocity[Parameter.COEFFICIENTS] /
                                            (np.sqrt(updated_squared_gradients[Parameter.COEFFICIENTS]) + epsilon))
        gradient[Parameter.INTERCEPTS] = (updated_velocity[Parameter.INTERCEPTS] /
                                            (np.sqrt(updated_squared_gradients[Parameter.INTERCEPTS]) + epsilon))
        SGD.update(self, p, gradient, id)
