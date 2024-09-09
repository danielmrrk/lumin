import numpy as np


class MiniBatchPreparer:
    def __init__(self, batches: int, X: np.array, y: np.array, num_samples: int):
        self.num_samples = num_samples
        self.batch_size = int(np.ceil(self.num_samples / batches))
        indices = np.random.permutation(num_samples)
        self.shuffled_X = X[indices]
        self.shuffled_y = y[indices]

    def get_batch(self, idx: int) -> tuple[np.array, np.array]:
        start_idx = idx * self.batch_size
        end_idx = min((idx + 1) * self.batch_size, self.num_samples)
        return self.shuffled_X[start_idx:end_idx], self.shuffled_y[start_idx:end_idx]
