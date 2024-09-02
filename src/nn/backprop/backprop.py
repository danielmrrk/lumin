import numpy as np
from PIL.DdsImagePlugin import module

from src.nn.module import Module


class Backprop:
    def __init__(self, module: Module, X: np.array, y: np.array):
        self.module = module
        self.X = X
        self.y = y

    def backprop(self):
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



