import Code_My_AI
import numpy as np


"""     Это пример, как можно использовать эту библиотеку, для аппроксимации сложной функции        """


# Создаём два экземпляра нейронки
ai = Code_My_AI.AI()
ai.name = "Sum_ai"

# Создаём архитектуру (сколько нейронов на каком слое)
ai.create_weights([2, 15, 15, 2], add_bias_neuron=True)

ai.what_act_func = ai.kit_act_funcs.Sigmoid

ai.number_disabled_weights = 0.0
ai.alpha = 1e-6

# Загружаем ии
# ai.load()


errors = []
learn_iteration = 0
while True:
    learn_iteration += 1

    # Наши данные
    data = [np.random.randint(10), np.random.randint(100)]

    # Ответ - это рандомная функция, которая принимает наши входные данные (data)
    answer = [data[0]**2 + 13, int(data[1]/5)]

    err = ai.learning(
        data,
        answer,
        get_error=True,
        type_error="regular",
        type_regularization="quadratic",
        regularization_value=1,
        regularization_coefficient=0.1,
    )

    if err is not None:  # Если функция вернула ошибку
        errors.append(err)

    if learn_iteration % (5_000) == 0:
        # Переводим от 0 до 1 в промежуток от min до max сложной функции
        ai_ans = ai.start_work(data) * 100
        ai_ans = [int(i) for i in ai_ans.tolist()]

        print("#", learn_iteration)
        print("Ответ:\t\t\t\t", answer)
        print("Результат нейронки: ", ai_ans)

        print("Общая ошибка:", int(np.mean(errors)))
        print(f"Сумма весов:", int(sum([np.sum(abs(i)) for i in ai.weights])))

        errors.clear()
        print()

        # Сохраняемся
        ai.delete()
        ai.save()
