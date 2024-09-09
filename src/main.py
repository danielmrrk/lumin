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

np.random.shuffle(data)

y_dev = data[0:1000, 0].astype(np.int32)
X_dev = data[0:1000, 1:].astype(np.float32) / 255

y_train = data[1000:7000, 0].astype(np.int32)
X_train = data[1000:7000, 1:].astype(np.float32) / 255

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
    optimizer=create_optimizer(OptimizerType.SGD)
)

model.fit(1000, 20, X_train, y_train)


def prediction(index):
    # Select the current image by its index
    current_image = X_train[index]

    # Reshape the image if it's flattened (assuming it's 28x28 for MNIST-like data)
    # If your image is already flattened, you can skip this step.
    current_image_flat = current_image.reshape(1, -1)  # Flatten to shape (1, 784) for prediction

    # Get the prediction for the specific image
    prediction = np.argmax(model.predict(current_image_flat), axis=1)

    # Get the true label for the image
    label = y_train[index]

    print("Prediction: ", prediction[0])  # [0] to extract the scalar value from the array
    print("Label: ", label)

    # Display the image in its original 2D form (28x28)
    current_image = current_image.reshape((28, 28))

    plt.gray()
    plt.imshow(current_image * 255, interpolation='nearest')  # Optional scaling back to 0-255
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




#y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
#y_dev = tf.keras.utils.to_categorical(y_dev, num_classes=10)
#
#model = tf.keras.Sequential([
#    tf.keras.layers.Dense(64, input_shape=(X_dev.shape[1],), activation='relu'),
#    tf.keras.layers.Dense(32, activation='relu'),
#    tf.keras.layers.Dense(16, activation='relu'),
#    tf.keras.layers.Dense(10, activation='softmax')
#])
#
#model.compile(
#    optimizer=tf.keras.optimizers.SGD(),
#    loss='categorical_crossentropy',
#    metrics=['accuracy']
#)
#
#batch_size = X_train.shape[0] // 20
#
#model.fit(X_train, y_train, epochs=200, batch_size=batch_size, verbose=1)
#
#y_pred = model.predict(X_dev)
#
#y_pred_labels = np.argmax(y_pred, axis=1)
#
#y_dev_labels = np.argmax(y_dev, axis=1)
#
#accuracy = accuracy_score(y_dev_labels, y_pred_labels)
# print("Accuracy:", accuracy)