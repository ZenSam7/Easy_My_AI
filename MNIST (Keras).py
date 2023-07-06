import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf


# Load the data and split it between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)


# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

model = keras.Sequential([
    keras.Input(shape=(28, 28, 1)),
    layers.Flatten(),

    layers.Dense(20, activation="relu"),
    layers.Dense(20, activation="relu"),

    layers.Dense(10, activation="softmax"),
])
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()


model.fit(x_train, y_train, batch_size=128, epochs=10, validation_split=0.1)


score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
