import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error  # Import scikit-learn's MSE function

from src.nn.activation.relu import ReLu
from src.nn.layer.layer import Layer
from src.nn.layer.output_layer import OutputLayer
from src.nn.layer.linear import Linear
from src.nn.loss.mse import MSE
from src.nn.module import Module
from src.nn.optimizer.sgd import SGD

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Generate synthetic dataset
X1 = 2 * np.random.rand(100, 1)  # Feature X1 in range [0, 2)
X2 = 3 * np.random.rand(100, 1)  # Feature X2 in range [0, 3)
X = np.hstack((X1, X2))

# Define the true relationship with multiple features
true_coefficients = np.array([5, -3]).reshape(-1, 1)
y = X @ true_coefficients + 4 + np.random.randn(100, 1) * 2  # Adding Gaussian noise

print(X.shape)
print(y.shape)



# Print the first few rows of the dataset
print("First few rows of the dataset (features and target):")
print(np.hstack((X, y))[:5])

# Define your custom model using your existing framework
m = Module(
    layers = [
        Layer(
            Linear(input_dim=X.shape[1], units=32),
            ReLu()
        ),
        Layer(
            Linear(32, units=16),
            ReLu()
        )],
    output_layer = OutputLayer(Linear(16, 1)),
    loss = MSE(),
    optimizer = SGD()
)

# Train the custom model
m.fit(300, 5, X=X, y=y)

# Define the TensorFlow model with the same architecture
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
model_tf.fit(X, y, epochs=300, batch_size=batch_size, verbose=1)

# Generate test data for predictions
X1_test = np.linspace(0, 2, 100)
X2_test = np.linspace(0, 3, 100)
X1_test_grid, X2_test_grid = np.meshgrid(X1_test, X2_test)
X_test = np.hstack((X1_test_grid.reshape(-1, 1), X2_test_grid.reshape(-1, 1)))

# Predict using the custom model
y_pred_custom = m.predict(X_test).reshape(-1)
# y_pred_custom = m.predict(X).reshape(-1)

# Predict using the TensorFlow model
y_pred_tf = model_tf.predict(X_test).reshape(-1)

# Compute the Mean Squared Error using scikit-learn
y_test = X_test @ true_coefficients + 4 + np.random.randn(10000, 1) * 2
mse_custom = mean_squared_error(y_test, y_pred_custom)
mse_tf = mean_squared_error(y_test, y_pred_tf)

# Print the MSE for both models
print("\nMean Squared Error for Custom Model Predictions:", mse_custom)
print("Mean Squared Error for TensorFlow Model Predictions:", mse_tf)

# Create the 3D scatter plot and line plot with Plotly
fig = go.Figure()

# Add the scatter plot for the training data
fig.add_trace(go.Scatter3d(
    x=X1.flatten(),
    y=X2.flatten(),
    z=y.flatten(),
    mode='markers',
    marker=dict(size=5, color='blue', opacity=0.5),
    name='Training Data'
))

# Add the surface plot for the custom model's predictions
fig.add_trace(go.Surface(
    x=X1_test_grid,
    y=X2_test_grid,
    z=y_pred_custom.reshape(X1_test_grid.shape),
    colorscale='Viridis',
    opacity=0.7,
    name='Predicted Surface (Custom Model)'
))

# Add the surface plot for the TensorFlow model's predictions
fig.add_trace(go.Surface(
    x=X1_test_grid,
    y=X2_test_grid,
    z=y_pred_tf.reshape(X1_test_grid.shape),
    colorscale='Cividis',
    opacity=0.7,
    name='Predicted Surface (TensorFlow Model)'
))

# Set plot titles and labels
fig.update_layout(
    title='Comparison of Custom Model and TensorFlow Model Predictions',
    scene=dict(
        xaxis_title='X1',
        yaxis_title='X2',
        zaxis_title='y'
    ),
    width=800,
    height=600
)

# Show the plot
fig.show()
