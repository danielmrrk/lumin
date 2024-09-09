from src.nn.optimizer.gradient_descent import GradientDescent

class SGD(GradientDescent):
    def __init__(self, lr: float = 1e-3):
        super().__init__(lr=lr)
        self.mini_batch = True
