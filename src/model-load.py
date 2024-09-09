import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.metrics import accuracy_score

data_df = pd.read_csv("../data/train.csv")
data = np.array(data_df)
n, m = data.shape

np.random.seed(42)

np.random.shuffle(data)

y_dev = data[0:1000, 0].astype(np.int32)
X_dev = data[0:1000, 1:].astype(np.float32) / 255

#y_train = data[1000:7000, 0].astype(np.int32)
#X_train = data[1000:7000, 1:].astype(np.float32) / 255

with open('../models/mnist-digits.pkl', 'rb') as file:
    model = pickle.load(file)

y_pred = model.predict(X_dev).reshape(X_dev.shape[0], 10)

def get_softmax_accuracy(y_pred, y):
    y_pred_labels = np.argmax(y_pred, axis=1)
    accuracy = accuracy_score(y, y_pred_labels)
    print("Accuracy:", accuracy)

def prediction(index, X, y):
    current_image = X[index]
    current_image_flat = current_image.reshape(1, -1)  # Flatten to shape (1, 784) for prediction
    prediction = np.argmax(model.predict(current_image_flat), axis=1)
    label = y[index]
    current_image = current_image.reshape((28, 28))
    plt.gray()
    plt.imshow(current_image * 255, interpolation='nearest')
    plt.title(f"Prediction: {prediction[0]}, Actual Label: {label}")
    plt.show()

#prediction(4)
#prediction(10)
#prediction(20)

get_softmax_accuracy(y_pred, y_dev)