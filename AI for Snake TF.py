import keras
import numpy as np
import tensorflow as tf
from keras.layers import (
    Flatten,
    Dense,
    Input,
)
from Games import Snake
from time import time
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

start = time()

# Создаём Змейку
snake = Snake(7, 5, amount_food=1, amount_walls=0,
              max_steps=60, display_game=False,
              dead_reward=-400, win_reward=200, cell_size=120)

tf.config.set_visible_devices(tf.config.list_physical_devices("CPU"), "CPU")

model = keras.Sequential([
    Dense(9, activation="tanh"),
    Dense(100, activation="tanh"),
    Dense(100, activation="tanh"),
    Dense(100, activation="tanh"),
    Dense(4, activation="softmax"),
])

model.compile(
        optimizer=tf.keras.optimizers.SGD(1e-3),  # "adam",
        loss="mean_absolute_error",
)
model.build((1, 9))

model.summary()

"""
       ДЛЯ ЗМЕЙКИ ИСПОЛЬЗОВАТЬ TF ВМЕСТО МОЕЙ БИБЛИОТЕКИ БЕССМЫСЛЕННО (да и для любых других простых задач)
                               т.к. рабоатет СИЛЬНО дольше, да и Q-обучения нету
              (его нет потому что этот метод обучения нельзя оптимизировать и выполнять на gpu)
"""

learn_iteration: int = 0
q_table = {(0): [0]}
future_state = (0)
while 1:
    learn_iteration += 1
    reward = 0

    if learn_iteration % 500 == 0:
        # Выводим максимальный и средний счёт змейки за 500 шагов
        max, mean = snake.get_max_mean_score()
        print(
            learn_iteration,
            "\t\tMax:", max,
            "\t\tMean:", round(mean, 1),
            "\t\t", int(time() - start), "s",
        )
        start = time()

    # Записываем данные в ответ
    state = tuple(snake.get_blocks(3))

    action_indx = np.argmax(model.predict(np.matrix(state), verbose=False))
    action = ("left", "right", "up", "down")[action_indx]
    reward = snake.step(action)

    # Если действие новое, то добавляем нули
    q_table.setdefault(state, [0, 0, 0, 0])

    # Формируем ответ
    answer = [0, 0, 0, 0]
    answer[np.argmax(q_table[state])] = 1

    # Обновляем Q-таблицу
    q_table[state][action_indx] = 0.9 * q_table[state][action_indx] + 0.1 * (reward + 0.6 * np.max(q_table[future_state]))

    future_state = state

    # Обучаем
    model.train_on_batch(np.matrix(state), np.matrix(answer))
