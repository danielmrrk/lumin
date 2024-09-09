import numpy as np

from src.nn.activation.activation import Activation


class Softmax(Activation):
    def __init__(self):
        super().__init__()
        self.__output: np.array = None

    def forward_train(self, z: np.array) -> np.array:
        self.__output =  self.forward(z)
        return self.__output

    def forward(self, z: np.array) -> np.array:
        exp = np.exp(z)
        exp_sum = np.sum(exp, axis=1, keepdims=True)
        return exp / exp_sum

    def backward_input(self, grad_output: np.array, kwargs: dict = None) -> np.array:
        # Compute the gradient of the loss (L) with respect to the logits (z) directly.
        # For softmax combined with categorical cross-entropy, the gradient simplifies to:
        # ∂L/∂z = softmax_output - one_hot_labels
        # This is because the derivative of the cross-entropy loss with respect to the logits
        # combines naturally with the softmax function, yielding this simplified form.
        if not kwargs:
            raise Exception("Softmax needs additional arguments for backprop.")
        y_indices = kwargs["y_indices"]

        examples = len(y_indices)
        hot_ones = np.zeros_like(self.__output)
        hot_ones[np.arange(examples), y_indices] = 1
        softmax_grad = self.__output - hot_ones
        return softmax_grad / examples