from src.nn.optimizer.optimizer import Optimizer
from src.utility.parameter import Parameters, Parameter

class GradientDescent(Optimizer):
    def __init__(self, lr: float = 1e-3):
        super().__init__(mini_batch=False)
        self.lr = lr

    def update(self, p: Parameters, gradient: Parameters):
        p[Parameter.COEFFICIENTS] -= self.lr * gradient[Parameter.COEFFICIENTS]
        p[Parameter.INTERCEPTS] -= self.lr * gradient[Parameter.INTERCEPTS]
