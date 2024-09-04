import numpy as np

class Backprop:
    def __init__(self, module, X: np.array, y: np.array):
        from src.nn.module import Module
        self.module: Module = module
        self.X = X
        self.y = y

    def backprop(self) -> float:
        input = self.X
        for layer in self.module.layers:
            input = layer.forward(input)

        y_hat = self.module.output_layer.forward(input)
        examples = len(self.y)

        # gradient in respect to the loss examples
        grad_output = (y_hat - self.y) / examples
        grad_output = self.module.output_layer.backprop(grad_output)

        for layer in self.module.layers[::-1]:
            grad_output = layer.backprop(grad_output)

        return np.sum((y_hat - self.y)**2)



