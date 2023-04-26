import Code_My_AI
import numpy as np
from time import time


start_time = time()

# Создаём экземпляр нейронки
ai = Code_My_AI.AI()


# Создаём архитектуру (сколько нейронов на каком слое)
ai.create_weights([2, 4, 1], add_bias_neuron=True)


ai.what_activation_function = ai.activation_function.ReLU
# ai.activation_function.value_range(-333, 1000)
# ai.end_activation_function = ai.activation_function.Sigmoid


ai.number_disabled_neurons = 0.0
ai.packet_size = 1
ai.alpha = 0.0000005



len_iterations_learning = 10000


errors = []
last_weights = [i.copy() for i in ai.weights]
for learn_iteration in range(1, len_iterations_learning +1):
    # Наши данные
    data = [np.random.randint(100), np.random.randint(100)]
    answer = [round(11* data[0] + 2.3* data[1] - 514, 1)]


    err = ai.learning(data, answer, True)

    if err != None:
        errors.append(err)


    # За всё обучение отображаем данные 10 раз
    if learn_iteration %  (0.1*len_iterations_learning) == 0:
        for i, j in zip(ai.weights, last_weights):
            print(i)
        print(ai.start_work(data, True)[1])
        last_weights = [i.copy() for i in ai.weights]


        print("#", learn_iteration)
        print("Ответ:", answer)
        print("Результат нейронки:", ai.start_work(data))
        print("Ошибка: %", round((ai.start_work(data)[0] - answer[0])\
                                 /answer[0]*100,1) )

        print("Ошибка:", int(np.mean(errors)))
        errors.clear()
        print()


print("Время на 1 итерацию:", format(round(
    (time() - start_time) / len_iterations_learning,
    10), '.10f'))


