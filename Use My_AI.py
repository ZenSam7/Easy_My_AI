import Code_My_AI
import numpy as np

# Создаём экземпляр нейронки
ai = Code_My_AI.AI()


# Создаём архитектуру (сколько нейронов на каком слое)
ai.create_weights([2, 3, 3, 1], add_bias_neuron=False)

ai.activation_function.value_range(0, 1)
ai.what_activation_function = ai.activation_function.ReLU
ai.end_activation_function = None



errors = []
for learn_iteration in range(1000):
    # Наши данные
    data = [np.random.randint(100), np.random.randint(100)]
    answer = [data[0] * data[1]]

    errors.append(ai.learning(data, answer, True))

    # Отображаем данные каждые __ итераций обучения
    if learn_iteration % 100 == 0:
        print("#", learn_iteration)
        print("Ответ:", answer)
        print("Результат нейронки:", ai.start_work(data))
        print("Ошибка: %", round((ai.start_work(data)[0] - answer[0]),1) )
        print("Ошибка: ", round(float(np.mean(errors)), 1) )
        errors = []
        print()


