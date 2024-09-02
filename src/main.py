from src.nn.linear import Linear
import numpy as np

from src.utility.parameter import Parameter

X = np.array([[1, 2], [2, 2], [3, 2], [4, 2], [5, 2]])
y = np.array([2, 4, 6, 8, 10])

np.random.seed(42)

l = Linear(X.shape[1], 4)
print(l.p[Parameter.COEFFICIENTS])
print(l.p[Parameter.INTERCEPTS])

output = l.forward(X)
print(output)

print(X.T)
