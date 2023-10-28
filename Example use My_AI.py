from My_AI import AI
import numpy as np


"""     Это пример, как можно использовать эту библиотеку, для аппроксимации сложной функции        """


# Создаём экземпляр нейронки
ai = AI()

# Создаём архитектуру (сколько нейронов на каком слое)
ai.create_weights([2, 10, 10, 10, 2], add_bias_neuron=True)

ai.what_act_func = ai.kit_act_func.ReLU
ai.end_act_func = ai.kit_act_func.ReLU_2


ai.number_disabled_weights = 0.0
ai.alpha = 1e-9


# Загружаем ии
# ai.load_data("Sum_ai")


errors = []
learn_iteration = 0
while 1:
    learn_iteration += 1


    # Наши данные
    data = [np.random.randint(100), np.random.randint(100)]

    # Ответ - это рандомная функция, которая принимает наши входные данные (data)
    answer = [int(11.03* data[0] + 2.23* data[1] + 729), data[0] + 133]


    err = ai.learning(data, answer, get_error=True)
    if err != None:   # Если функция вернула ошибку
        errors.append(err)


    if learn_iteration % (2_000) == 0:
        ai_ans = [int(i) for i in ai.predict(data).tolist()]
        # print(sum([np.sum(i) for i in ai.weights]))

        print("#", learn_iteration)
        print("Ответ: \t \t \t \t", answer)
        print("Результат нейронки: ", ai_ans)
        print("Ошибка в этом примере: %", round(sum(ai_ans) / sum(answer) -1, 2))

        print("Общая ошибка:", int(np.mean(errors)))
        errors.clear()
        print()

        # Сохраняемся
        ai.delete_data("Sum_ai")
        ai.save_data("Sum_ai")