import numpy as np

from src.nn.activation.activation import Activation
from src.nn.activation.softmax import Softmax
from src.nn.loss.loss import Loss
from src.nn.optimizer.optimizer import Optimizer


def backprop(module, X: np.array, y: np.array, loss: Loss, optimizer: Optimizer, output_activation: Activation) -> float:
    """
    Perform backpropagation to update the model's weights based on the provided loss and optimizer.

    This function performs a forward pass through the module's layers, computes the loss,
    and then performs a backward pass to compute gradients and update weights.

    Args:
        module (Module): The neural network module containing layers and an output layer.
        X (np.array): The input data for training.
        y (np.array): The true labels corresponding to the input data.
        loss (Loss): The loss function to compute the loss and its gradient.
        optimizer (Optimizer): The optimizer used to update the model's parameters.
        output_activation (ActivationType): The activation can sometimes be needed if the activation function needs
        to pass additional arguments for gradient calculation.

    Returns:
        float: The average loss over the batch.
        :param output_activation:
    """
    from src.nn.module import Module
    module: Module = module

    input = X
    for layer in module.layers:
        input = layer.forward_train(input)

    y_hat = module.output_layer.forward_train(input)

    grad_output = loss.gradient(y_hat, y)

    kwargs = {}
    if isinstance(output_activation, Softmax):
        kwargs = {"y_indices": y}

    grad_output = module.output_layer.backprop(grad_output, optimizer, kwargs)

    for layer in module.layers[::-1]:
        grad_output = layer.backprop(grad_output, optimizer)

    return loss.loss(y_hat, y)


