import Code_My_AI
import numpy as np
from time import time


start_time = time()

# Создаём экземпляр нейронки
ai = Code_My_AI.AI()


# Создаём архитектуру (сколько нейронов на каком слое)
ai.create_weights([2, 4, 2], add_bias_neuron=False)


ai.what_activation_function = ai.activation_function.ReLU
# ai.activation_function.value_range(-100, 100)
# ai.end_activation_function = ai.activation_function.Sigmoid


ai.number_disabled_neurons = 0.0
ai.packet_size = 1
ai.alpha = 0.000001

# ai.save_data("Sum_ai")
# ai.delete_data("My_ai")
# ai.load_data("Sum_ai")

len_iterations_learning = 10000



errors = []
for learn_iteration in range(1, len_iterations_learning +1):
    # Наши данные
    data = [np.random.randint(100), np.random.randint(100)]
    answer = [round(11* data[0] + 2.3* data[1] -33, 1), 1]


    err = ai.learning(data, answer, True)

    if err != None:
        errors.append(err)


    # За всё обучение отображаем данные 10 раз
    if learn_iteration %  (0.1*len_iterations_learning) == 0:

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

