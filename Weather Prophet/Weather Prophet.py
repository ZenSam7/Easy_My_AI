from keras import Sequential
from keras.layers import Flatten, Dropout, Dense, LSTM
import tensorflow as tf

from datetime import datetime
from time import time
import numpy as np

# Убираем предупреждения
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import logging
tf.get_logger().setLevel(logging.ERROR)


"""
    Example Weather Dataset:
    05.07.2023 15:00;24,7;749,8;762,7;56;1;70;;;;
    ...

    In this string:

    05.07.2023 15:00 | 24,7        | 749,8               | 56                | 70
    Data & Time      | Temperature | Atmosphere Pressure | Relative Humidity | Cloudiness
                     |  ℃         |  mm Hg (on station) | %                 | %
"""


chars_in_progress_bar = 40


DATA = []
for NAME_DATASET in ["Москва (ВДНХ)", "Москва (Аэропорт)", "Москва (Центр)"]:
    print("LOADING DATASET:", NAME_DATASET)
    print("|", end="")
    with open(f"Datasets/{NAME_DATASET}.csv") as dataset:
        # Без первой строки с начальным символом;
        # Без первой (идём от староого к новому) записи, т.к. она используется для смещения ответа
        # (т.е. чтобы мы на основе предыдущей записи создавали следующую)
        records = dataset.readlines()[1:][::-1]
        len_file = len(records)
        num_data = 0

        for string in records:
            data = string.split(";")[:-2]

            # Если попался брак, то пропускаем шаг
            if '' in data or len(data) != 5:
                continue


            # Преобразуем строку
            # data[0] -> hours (beginning with new year)
            data[0] = int(datetime.strptime(data[0], "%d.%m.%Y %H:%M").timestamp() // (60*60) %(24*365))
            data[1] = float(data[1].replace(",", "."))
            data[2] = float(data[2].replace(",", "."))
            data[3] = int(data[3])
            data[4] = int(data[4])

            DATA.append(data)
            num_data += 1

            # Типа progress bar
            if num_data % (len_file // chars_in_progress_bar) == 0:
                print("#", end="")

    print("|\n")


"""Создаём ИИшку"""
ai = Sequential([

    LSTM(100, activation="relu", return_sequences=True, unroll=True),
    LSTM(100, activation="relu", return_sequences=True, unroll=True),
    LSTM(100, activation="relu", return_sequences=True, unroll=True),

    Dense(5, activation="linear"),
])
ai.build(input_shape=(num_data, 1, 5))
ai.compile(optimizer="adam", loss="mean_squared_error")
# ai.summary()


"""Сохранения / Загрузки"""
save_path = lambda name: "Saves Weather Prophet/{name}".format(name=name)

# Как загружать:
ai = tf.keras.models.load_model(save_path("first_save"))

# Как сохранять:
# ai.save(save_path("first_save"))



"""Обучение"""
# DATA_with_bias == Данные погоды, начинавя с 1ого дня (принимает)
# DATA == Данные погоды, начинавя с 2ого дня (должен предсказать)

# Создаём смещени назад во времени при помощи первой записи из "Москва (ВДНХ)"
DATA_with_bias = [[981, -9.1, 751.0, 85, 100]] + DATA[:-1]

DATA = np.array(DATA).reshape((len(DATA), 1, 5))
DATA_with_bias = np.array(DATA_with_bias).reshape((len(DATA_with_bias), 1, 5))

# DATA_with_bias == Данные погоды, начинавя с 1ого дня (принимает)
# DATA == Данные погоды, начинавя с 2ого дня (должен предсказать)
ai.fit(DATA_with_bias, DATA, epochs=2, batch_size=100, verbose=True, shuffle=False)


# Сохраняем
ai.save(save_path("first_save"))
print("Save")
