import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.metrics import accuracy_score
from src.nn.layer.layer import Layer
from src.nn.layer.linear import Linear
from src.nn.layer.output_layer import OutputLayer
from src.nn.module import Module
from src.nn.optimizer.optimizer_factory import create_optimizer
from src.nn.type import ActivationType, NormalizationType, LossType, OptimizerType

data_df = pd.read_csv("../data/train.csv")
data = np.array(data_df)
n, m = data.shape

print(n)

np.random.seed(42)

np.random.shuffle(data)

y_dev = data[0:1000, 0].astype(np.int32)
X_dev = data[0:1000, 1:].astype(np.float32) / 255

y_train = data[1000:n, 0].astype(np.int32)
X_train = data[1000:n, 1:].astype(np.float32) / 255

model = Module(
    layers = [
        Layer(
            Linear(input_dim=X_dev.shape[1], units=64),
            ActivationType.RELU
        ),
        Layer(
            Linear(input_dim=64, units=32),
            ActivationType.RELU
        ),
        Layer(
            Linear(input_dim=32, units=16),
            ActivationType.RELU
        )
    ],
    output_layer=OutputLayer(
            Linear(input_dim=16, units=10),
            ActivationType.SOFTMAX,
            NormalizationType.MAX
        ),
    loss_type=LossType.CROSS_ENTROPY,
    optimizer=create_optimizer(OptimizerType.ADAM)
)

model.fit(80, 120, X_train, y_train)


def prediction(index):
    current_image = X_train[index]
    current_image_flat = current_image.reshape(1, -1)  # Flatten to shape (1, 784) for prediction
    prediction = np.argmax(model.predict(current_image_flat), axis=1)
    label = y_train[index]
    current_image = current_image.reshape((28, 28))
    plt.gray()
    plt.imshow(current_image * 255, interpolation='nearest')
    plt.title(f"Prediction: {prediction[0]}, Actual Label: {label}")
    plt.show()

prediction(4)
prediction(10)
prediction(20)

y_pred = model.predict(X_train).reshape(X_train.shape[0], 10)

def get_softmax_accuracy(y_pred, y):
    y_pred_labels = np.argmax(y_pred, axis=1)
    accuracy = accuracy_score(y, y_pred_labels)
    print("Accuracy:", accuracy)

get_softmax_accuracy(y_pred, y_train)

answer = input("Do you want to updated mnist-digits?")
if answer == "yes":
    with open('../models/mnist-digits.pkl', 'wb') as file:
        pickle.dump(model, file)

