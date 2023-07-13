from keras import Sequential
import keras
from keras.layers import Flatten, Dense, SimpleRNN, LSTM, BatchNormalization, Conv1D
import tensorflow as tf

from time import time
import numpy as np

# Убираем предупреждения
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
tf.get_logger().setLevel(logging.ERROR)

# # Работаем с GPU
# tf.config.set_visible_devices(tf.config.list_physical_devices('GPU'), 'GPU')
# Работаем с CPU
tf.config.set_visible_devices([], 'GPU')


chars_in_progress_bar = 34


DATA = []
for NAME_DATASET in ["Москва (ВДНХ)", "Москва (Центр)", "Москва (Аэропорт)"]:
    print(">>> LOADING DATASET:", NAME_DATASET)
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
            processed_data = [0 for _ in range(6)]

            # Преобразуем строку
            # data[0] -> часы (в течении дня)
            # data[1] -> месяц
            processed_data[0] = int(data[0][11:13])
            processed_data[1] = int(data[0][3:5])
            processed_data[2] = float(data[1].replace(",", "."))
            processed_data[3] = float(data[2].replace(",", "."))
            processed_data[4] = int(data[3])
            processed_data[5] = int(data[4])

            DATA.append(processed_data)

            # Типа progress bar
            num_data += 1
            if num_data % (len_file // chars_in_progress_bar) == 0:
                print("#", end="")

    print("|")
print("\n")

"""
0) Температура
1) Давление
2) Влажность
3) Облачность
"""



"""Создаём ИИшки"""
# Суть в том, чтобы расперелить задачи по предсказыванию между разными нейронками
# Т.к. одна нейросеть очень плохо предскаывает одновременно все факторы

# У всех нейронок одна архитектура и один вход
input_layer = keras.Input((1, 6))


def get_ai(name):
    model = Sequential([
        Conv1D(16, 6, padding="same"),
        BatchNormalization(),
        Conv1D(32, 6, padding="same"),
        BatchNormalization(),

        Dense(32, activation="relu"),
        BatchNormalization(),
        LSTM(32, return_sequences=True, unroll=True),
        BatchNormalization(),
        Dense(32, activation="relu"),
        BatchNormalization(),
        Dense(4, activation="relu"),

        Dense(1, activation="linear"),
    ])(input_layer)

    output = Dense(1, activation="linear", name=name)(model)

    return output


temperature = get_ai("temperature")
pressure = get_ai("pressure")
humidity = get_ai("humidity")
cloudiness = get_ai("cloudiness")


ai = keras.Model(input_layer, [temperature, pressure, humidity, cloudiness])
ai.compile(optimizer="adam", loss="mean_squared_error",
           loss_weights={"temperature": 100.0, "pressure": 1.0, "humidity": 10.0, "cloudiness": 1.0})
           # Отдаём приоритет температуре и влажности


"""Сохранения / Загрузки"""
def save_path(name): return "Saves Weather Prophet/{name}".format(name=name)


SAVE_NAME = "third_save"

# Как загружать: ai = tf.keras.models.load_model(save_path(AI_NAME))
# Как сохранять: ai.save(save_path(AI_NAME))

# ЗАГРУЖАЕМСЯ
# print(f"Loading the {SAVE_NAME}.", end="\t\t")
# ai = tf.keras.models.load_model(save_path(SAVE_NAME))
# print("Done\n")



"""DATA_with_bias == Данные погоды, начиная с 1ого дня (принимает)
   DATA == Данные погоды, начиная с 2ого дня (должен предсказать)"""

# Создаём смещени назад во времени при помощи первой записи из "Москва (ВДНХ)"
DATA_with_bias = [[0, 2, -9.1, 751.0, 85, 100]] + DATA[:-1]

DATA = np.array(DATA).reshape((len(DATA), 1, 6))
DATA_with_bias = np.array(DATA_with_bias).reshape((len(DATA_with_bias), 1, 6))

DATA = DATA - DATA_with_bias   # Остаточное обучение
DATA = DATA[:, :, 2:]          # (ИИшке не надо предсказывать время)


"""Обучение"""

callbacks = [
    keras.callbacks.EarlyStopping(monitor="loss", min_delta=0, patience=5, verbose=False),
]

print(f">>> Learning the {SAVE_NAME}")

# Разделяем часть для обучения и для тестирования (Всего 133_066 записей)
# В качестве ответа записываем значение природного явления
train_data = DATA_with_bias[:-20_000]
train_data_answer = np.reshape(np.array([ DATA[:-20_000, 0, :] ]), (len(train_data), 1, 4))

test_data = DATA_with_bias[-20_000:]
test_data_answer = np.reshape(np.array([ DATA[-20_000:, 0, :] ]), (20_000, 1, 4))


ai.fit(train_data, train_data_answer, epochs=25, batch_size=100, verbose=True,
       shuffle=False, callbacks=callbacks)

print(">>> Testing:")
ai.evaluate(test_data, test_data_answer, batch_size=100, verbose=True)

print("\n")



# Сохраняем
print(f">>> Saving the {SAVE_NAME}.", end="  ")
ai.save(save_path(SAVE_NAME))
print("Done (Ignore the WARNING)")



# Отображаем предсказания ИИшек, и правильные ответ
# Создаём последовательность предсказаний ии, а потом сравниваем в реальными данными
sequence_len = 10

real_data, ai_pred = [], []
rand = np.random.randint(1, 20_000)

# Проверяем на данных, на которых они не обучались
for data in DATA_with_bias[-20_000:][rand : rand +sequence_len]:
    real_data.append(data.tolist()[0])

    ai_predict = [i[0,0,0] for i in ai.predict( np.resize(data, (1,1,6)), verbose=False)]
    ai_predict = [round(ai_predict[i] + real_data[-1][i +2], 1) for i in range(4)]

    ai_pred.append(ai_predict)

print("Time\t\t\t\t\tReal Data \t\t\t\t\t\t Ai Predict")
for real, pred in zip(real_data, ai_pred):
    print(real[:2], " \t\t ", real[2:],  " \t ", pred)

