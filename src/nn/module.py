import numpy as np

from src.nn.backprop.backprop import Backprop
from src.nn.layer.layer import Layer
from src.nn.layer.output_layer import OutputLayer


class Module:
    def __init__(self, layers: list[Layer], output_layer: OutputLayer):
        self.layers = layers
        self.output_layer = output_layer

    def fit(self, epochs: int, batches: int, X: np.array, y: np.array):
        num_samples = len(X)
        batch_size = int(np.ceil(num_samples / batches))

        for epoch in range(epochs):
            indices = np.random.permutation(num_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            total_loss = 0  # Accumulate the total loss for this epoch
            total_examples = 0  # Track the total number of examples processed

            for start_idx in range(0, num_samples, batch_size):
                end_idx = min(start_idx + batch_size, num_samples)
                batch_X = X_shuffled[start_idx:end_idx]
                batch_y = y_shuffled[start_idx:end_idx]

                # Calculate the loss for the current batch
                batch_loss = Backprop(self, batch_X, batch_y).backprop()

                # Accumulate total loss and count examples
                total_loss += batch_loss
                total_examples += len(batch_y)

                current_batch = (start_idx // batch_size) + 1

                # Optionally print the batch loss
                print(f"\rEpoch: {epoch + 1}/{epochs}, Batch: {current_batch}/{batches}", end="")

            # Average the total loss over all examples in the epoch
            avg_loss = total_loss / total_examples
            print(f"\nEpoch: {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}")

        print("Training complete.")

    def predict(self, X: np.array) -> np.array:
        input = X
        for layer in self.layers:
            input = layer.forward(input)

        return self.output_layer.forward(input)
