from keras import Sequential
from keras.layers import Flatten, Dense, SimpleRNN
import tensorflow as tf

from time import time
import numpy as np

# Убираем предупреждения
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import logging
tf.get_logger().setLevel(logging.ERROR)


chars_in_progress_bar = 30

DATA = []
for NAME_DATASET in ["Москва (ВДНХ)", "Москва (Центр)", "Москва (Аэропорт)"]:
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

    print("|\n")

"""
    Example Weather Dataset:
    05.07.2023 15:00;24,7;749,8;762,7;56;1;70;;;;
    ...

    In this string:

    05.07.2023 15:00  | 24,7        | 749,8               | 56                | 70
    Data & Time       | Temperature | Atmosphere Pressure | Relative Humidity | Cloudiness
    Convert to months |  ℃         |  mm Hg (on station) | %                 | %
    (starting from    |
    the beginning of  |
    the year)         |
"""



"""Создаём ИИшки"""
# Суть в том, чтобы расперелить задачи по предсказыванию между разными нейронками
# Т.к. одна нейросеть очень плохо предскаывает одновременно все факторы
AIs = []

for name in ["temperature", "pressure", "humidity", "cloudiness"]:
    ai = Sequential([
        SimpleRNN(50, activation="linear", return_sequences=True, unroll=True),
        Dense(1, activation="linear"),
    ])
    ai.build(input_shape=(num_data, 1, 6))
    ai.compile(optimizer="adam", loss="mean_squared_error")

    AIs.append(ai)



"""Сохранения / Загрузки"""
save_path = lambda name: "Saves Weather Prophet/{name}".format(name=name)

# Как загружать: ai = tf.keras.models.load_model(save_path(AI_NAME))
# Как сохранять: ai.save(save_path(AI_NAME))

# ЗАГРУЖАЕМСЯ
AIs = []
for name in ["Temperature", "Pressure", "Humidity", "Cloudiness"]:
    print(f"Loading the {name} ai.", end=" ")
    AIs.append(tf.keras.models.load_model(save_path(name)))
    print("\tDone")
print("\n")



"""DATA_with_bias == Данные погоды, начиная с 1ого дня (принимает)
   DATA == Данные погоды, начиная с 2ого дня (должен предсказать)"""

# Создаём смещени назад во времени при помощи первой записи из "Москва (ВДНХ)"
DATA_with_bias = [[0, 2, -9.1, 751.0, 85, 100]] + DATA[:-1]

DATA = np.array(DATA).reshape((len(DATA), 1, 6))
DATA_with_bias = np.array(DATA_with_bias).reshape((len(DATA_with_bias), 1, 6))


# Отображаем предсказания ИИшек, и правильные ответ
# for _ in range(5):
#     rand = np.random.randint(1, 10_000)
#
#     print("INPUT DATA:\t", DATA_with_bias[rand][0, 2:].tolist())
#     print("AI ANSWER:\t", [round(ai.predict( np.array([DATA_with_bias[rand]])
#                                        , verbose=False)[0,0].tolist()[0], 1)  for ai in AIs])
#     print("ANSWER:\t\t", DATA[rand][0, 2:].tolist())
#     print()



"""
0) Время (часы с начала дня)
1) Время (месяц)
2) Температура
3) Давление
4) Влажность
5) Облачность
"""

"""Обучение"""
for ai, name, index in zip(AIs, ["Temperature", "Pressure", "Humidity", "Cloudiness"], [2, 3, 4, 5]):
    print(f">>> Learning the {name} ai")

    # Разделяем часть для обучения и для тестирования (Всего 133_066 записей)
    # В качестве ответа записываем значение природного явления
    train_data = DATA_with_bias[:-10000]
    train_data_answer = np.reshape(np.array([ DATA[:-10000, 0, index] ]), (len(train_data), 1, 1))

    test_data = DATA_with_bias[-10000:]
    test_data_answer = np.reshape(np.array([ DATA[-10000:, 0, index] ]), (len(test_data), 1, 1))


    ai.fit(train_data, train_data_answer, epochs=5, batch_size=50, verbose=True, shuffle=False)

    print(">>> Testing:")
    ai.evaluate(test_data, test_data_answer, batch_size=100, verbose=True)

    print("\n")



# Сохраняем
for ai, name in zip(AIs, ["Temperature", "Pressure", "Humidity", "Cloudiness"]):
    print(f"Saving the {name} nn", end=" ")
    ai.save(save_path(name))
    print("Done")
