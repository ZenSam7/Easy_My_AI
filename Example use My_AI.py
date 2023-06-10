import Code_My_AI
import numpy as np


"""     Это пример, как можно использовать эту библиотеку, для аппроксимации сложной функции        """


# Создаём экземпляр нейронки
ai = Code_My_AI.AI()

# Создаём архитектуру (сколько нейронов на каком слое)
ai.create_weights([2, 10, 2], add_bias_neuron=True)

ai.what_act_func = ai.act_func.ReLU_2
ai.act_func.value_range(-100, 333)
ai.end_act_func  = ai.act_func.ReLU_2


ai.number_disabled_neurons = 0.0
ai.alpha = 1e-7


# Загружаем ии
# ai.load_data("Sum_ai")


errors = []
learn_iteration = 0
while 1:
    learn_iteration += 1


    # Наши данные
    data = [np.random.randint(100), np.random.randint(100)]

    # Ответ - это рандомная функция, которая принимает наши входные данные (data)
    answer = [data[0] - data[1] +13, int(3.22* data[0])]


    err = ai.learning(data, answer, get_error=True, type_error=1, regularization=1, regularization_value=100)
    if err != None:   # Если функция вернула ошибку
        errors.append(err)


    if learn_iteration % (1_000) == 0:
        ai_ans = [int(i) for i in ai.start_work(data).tolist()]
        print(sum([np.sum(abs(i)) for i in ai.weights]))

        print("#", learn_iteration)
        print("Ответ: \t \t \t \t", answer)
        print("Результат нейронки: ", ai_ans)
        print("Ошибка в этом примере: %", round(sum(ai_ans) / (sum(answer)+0.1) -1, 2))

        print("Общая ошибка:", int(np.mean(errors)))
        errors.clear()
        print()

        # Сохраняемся
        ai.delete_data("Sum_ai")
        ai.save_data("Sum_ai")