import numpy as np
from tqdm import tqdm

from src.nn.backprop.backprop import backprop
from src.nn.data_preparation.mini_batch_preparer import MiniBatchPreparer
from src.nn.layer.layer import Layer
from src.nn.layer.output_layer import OutputLayer
from src.nn.loss.loss import Loss
from src.nn.loss.loss_factory import loss_factory
from src.nn.optimizer.optimizer import Optimizer
from src.nn.optimizer.optimizer_factory import optimizer_factory
from src.nn.type import LossType, OptimizerType


class Module:
    def __init__(self, layers: list[Layer], output_layer: OutputLayer, loss_type: LossType, optimizer: Optimizer):
        self.layers = layers
        self.output_layer: OutputLayer = output_layer
        self.loss: Loss = loss_factory[loss_type]()
        self.optimizer = optimizer

    def fit(self, epochs: int, batches: int, X: np.array, y: np.array, verbose: bool = True):
        num_samples = len(y)
        batches = batches if self.optimizer.mini_batch else 1

        for epoch in range(epochs):
            mini_batch_preparer = MiniBatchPreparer(batches, X, y, num_samples)

            total_loss = 0

            for idx in range(batches):
                batch_X, batch_y = mini_batch_preparer.get_batch(idx)
                batch_loss = backprop(self, batch_X, batch_y, self.loss, self.optimizer)

                total_loss += batch_loss * len(batch_y)

                print(f"\rEpoch: {epoch + 1}/{epochs}, Batch: {idx + 1}/{batches}, Batch Loss: {batch_loss:.4f}", end="")

            avg_loss = total_loss / num_samples
            print(f"\nEpoch: {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}")

        print("Training complete.")

    def predict(self, X: np.array) -> np.array:
        input = X
        for layer in self.layers:
            input = layer.forward(input)

        return self.output_layer.forward(input)
