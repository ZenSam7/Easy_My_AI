import Code_My_AI
import numpy as np

# Создаём экземпляр нейронки
ai = Code_My_AI.AI()


# Создаём архитектуру (сколько нейронов на каком слое)
ai.create_weights([2, 22, 22, 22, 1])

# Указываем, какую функцию активации будем использовать (можно и свою придумать)
ai.what_activation_function = ai.activation_function.ISRU


# print("Веса:")
# for i in ai.matrix_weights:
#     print(i)
# print()


data = [44, 1]

print("Результат работы нейронки:")
print(ai.start_work(data))
print()

ai.save_data("first ai")

