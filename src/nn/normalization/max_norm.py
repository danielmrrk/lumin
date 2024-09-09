import numpy as np

def max_norm(z: np.array):
    max_z = np.max(z, axis=1, keepdims=True)
    return z - max_z
