import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from src.nn.optimizer.gradient_descent import GradientDescent
from src.nn.optimizer.optimizer_factory import create_optimizer
from src.nn.type import ActivationType, LossType, OptimizerType

# Generate 1D data with a square root shape and add some noise
np.random.seed(42)  # For reproducibility
X = np.linspace(0, 10, 100).reshape(-1, 1)  # 100 samples, 1 feature
y = np.sqrt(X) + np.random.normal(0, 0.1, X.shape)  # Square root shape with noise

print(X.shape)
print(y.shape)



# Visualize the generated data
plt.scatter(X, y, label='Data with noise')
plt.title('Generated Data with Square Root Shape and Noise')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()

# Import your custom module classes
from src.nn.activation.relu import ReLu
from src.nn.layer.layer import Layer
from src.nn.layer.output_layer import OutputLayer
from src.nn.layer.linear import Linear
from src.nn.loss.mse import MSE
from src.nn.module import Module
from src.nn.optimizer.sgd import SGD

# Initialize and train your custom model
m = Module(
    layers=[
        Layer(
            Linear(input_dim=X.shape[1], units=32),
            ActivationType.RELU
        ),
        Layer(
            Linear(32, units=16),
            ActivationType.RELU
        )
    ],
    output_layer=OutputLayer(Linear(16, 1)),
    loss_type=LossType.MSE,
    optimizer=create_optimizer(OptimizerType.GRADIENT_DESCENT, lr=1e-3),
)

m.fit(epochs=300, batches=5, X=X, y=y)

# Initialize and train the TensorFlow model
model_tf = Sequential([
    Dense(32, input_dim=X.shape[1], activation='relu'),
    Dense(16, activation='relu'),
    Dense(1)  # Output layer for regression
])

# Compile the TensorFlow model
model_tf.compile(optimizer='sgd', loss='mean_squared_error')

# Train the TensorFlow model
number_of_batches_per_epoch = 5
number_of_samples = X.shape[0]
batch_size = number_of_samples // number_of_batches_per_epoch
model_tf.fit(X, y, epochs=100, batch_size=batch_size, verbose=1)

# Predict with both models
y_pred_tf = model_tf.predict(X).flatten()
y_pred_custom = m.predict(X)

# Compute mean squared error for both models
mse_tf = mean_squared_error(y, y_pred_tf)
mse_custom = mean_squared_error(y, y_pred_custom)

print(f"TensorFlow Model MSE: {mse_tf:.4f}")
print(f"Custom Model MSE: {mse_custom:.4f}")

# Plot the results using Matplotlib
plt.figure(figsize=(10, 6))
plt.scatter(X, y, label='True Data', alpha=0.6)
plt.plot(X, y_pred_tf, label='TensorFlow Model Predictions', color='r', linewidth=2)
plt.plot(X, y_pred_custom, label='Custom Model Predictions', color='g', linestyle='--', linewidth=2)
plt.title('Model Predictions vs True Data')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
