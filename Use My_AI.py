import Code_My_AI
import numpy as np

# Создаём экземпляр нейронки
ai = Code_My_AI.AI()


# Создаём архитектуру (сколько нейронов на каком слое)
ai.create_weights([1, 2, 1], add_bias_neuron=True)



data = [42]


ai.activation_function.value_range(0, 20)
ai.end_activation_function = ai.activation_function.Sigmoid


for i in ai.weights:
    print(i)
print(ai.start_work(data, True))

# for learn_iteration in range(100):
#     # Наши данные
#     data = [np.random.randint(10), np.random.randint(10)]
#
#     print("#" + str(learn_iteration))
#     print("Ответ: ", sum(data))
#     print("Результат нейронки: ", ai.start_work(data)[0])
#     print()
#     ai.learning(data, [sum(data)])