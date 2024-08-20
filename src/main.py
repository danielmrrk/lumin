import numpy as np

from src.model.linear_regression import LinearRegression
from sklearn.linear_model import LinearRegression as LR
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor


model = LinearRegression(alpha=1e-2, iter=20000)
test_model = SGDRegressor(learning_rate='constant', eta0=1e-2, max_iter=10000)

X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

model.fit(X, y)
test_model.fit(X, y)
y_hat = model.predict(X)
y_test = test_model.predict(X)
print(model.parameters)

plt.scatter(X, y, color="red")
plt.plot(X, y_hat, color="green")
plt.plot(X, y_test, color="blue")
plt.show()
