import numpy as np

from src.nn.loss.loss import Loss
from src.nn.optimizer.optimizer import Optimizer


def backprop(module, X: np.array, y: np.array, loss: Loss, optimizer: Optimizer) -> float:
    from src.nn.module import Module
    module: Module = module

    input = X
    for layer in module.layers:
        input = layer.forward_train(input)

    y_hat = module.output_layer.forward_train(input)

    grad_output = loss.gradient(y_hat, y)
    grad_output = module.output_layer.backprop(grad_output, optimizer)

    for layer in module.layers[::-1]:
        grad_output = layer.backprop(grad_output, optimizer)

    return loss.loss(y_hat, y) / len(y)



