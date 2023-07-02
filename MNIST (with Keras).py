import numpy as np
import matplotlib.pyplot as mpl
from mnist import MNIST
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
print(">>> Start")

mnist = MNIST()


"""Архитектура"""
ai = keras.Sequential([
    # Входы
    # Conv2D(20, (3, 3), padding="same",  activation="relu", input_shape=(28, 28)),
    # MaxPooling2D((2, 2), strides=2),
    # Conv2D(20, (3, 3), padding="same",  activation="relu"),
    # MaxPooling2D((2, 2), strides=2),
    # Flatten(),

    Flatten(input_shape=(784,)),

    # Слои
    Dropout(0.2),
    Dense(60, activation="relu"),
    Dropout(0.2),
    Dense(60, activation="relu"),

    # Выход
    Dense(10, activation="softmax"),
])
ai.compile(
    optimizer="adam",          #keras.optimizers.Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)


"""Обучение"""
errors = []
for images, labels in mnist.train_set.minibatches(batch_size=60_000):
    History = ai.fit(images, labels, batch_size=2000, epochs=7, verbose=True)
    errors.extend(History.history["loss"])

mpl.figure(figsize=(7.5, 3), dpi=100).subplots_adjust(**{"left":0.045, "bottom":0.085, "right": 0.992, "top":0.990})
mpl.plot(errors)
mpl.grid(True)
mpl.show()

# print(ai.summary())


"""Тестирование"""
for images, labels in mnist.test_set.minibatches(batch_size=10_000):
    ai.evaluate(images, labels, verbose=True)
