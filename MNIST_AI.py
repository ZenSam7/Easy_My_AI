import Code_My_AI
from mnist import MNIST
import numpy as np
from time import time

start_time = time()

mnist = MNIST()

ai = Code_My_AI.AI()

ai.create_weights([784, 20, 20, 10], add_bias_neuron=True)

ai.what_act_func = ai.act_func.Sigmoid
ai.end_act_func = ai.act_func.Sigmoid

ai.number_disabled_weights = 0.2
ai.batch_size = 10
ai.alpha = 1e-4


# alpha:   1e-8   |   1e-6   |    1e-3
#          ⨉⨉⨉       ⨉⨉       !✓!✓!✓!
# regularization_coefficient:    0.0   |   0.1   |   0.3
#                                 ✓        ⨉?       ⨉

# Слоёв:   5-6  |   0-2
#           ?        ✓
# Слои по:  40-50   |   10-20    нейронов
#             ?           ✓

# add_bias_neuron: True   |   False
#                   ?           ?
# ReLU_2  |  Sigmoid
#   ✓           ✓
# number_disabled_weights:  0.0  |  0.3   |   0.5
#                            ✓?     ✓✓        ✓?
# type_error:       1    |     2    |     3
#                   ✓         ⨉⨉        ✓✓✓
# batch_size:     1      |     10
#                 ✓            ✓?


name = "MNIST"
# ai.save_data(name)
# ai.load_data(name)

ai.print_how_many_parameters()

print("Обучение...")

for cycle in range(1):
    print(f"Цикл #{cycle}")

    num, errors = 0, 0
    max_train_images, show_progress = 60_000, 0.10

    for images, labels in mnist.train_set.minibatches(batch_size=1):
        num += 1

        image = images.tolist()[0]
        label = labels.tolist()[0]
        ai.learning(
            image,
            label,
            type_error=1,
            type_regularization=1,
            regularization_value=10,
            regularization_coefficient=0.0,
            impulse_coefficient=0.9,
        )

        if np.argmax(ai.start_work(image)) != np.argmax(np.array(label)):
            errors += 1

        if num % int(max_train_images * show_progress) == 0:
            print(
                f">>> {int(num / max_train_images *100)}% \t\t",
                f"Images: {num} \t\t",
                f"Error: {round((errors / num) *100, 1)}% \t\t",
                f"Summ weights: {int( sum([np.sum(np.abs(i)) for i in ai.weights]) )}",
            )

            errors = 0

            # # Сохраняемся
            ai.delete(name)
            ai.save(name)

        if num == max_train_images:
            break

    print()


print("Тестирование")
num, accuracy = 0, 0
max_test_images = 6_000

for images, labels in mnist.test_set.minibatches(batch_size=1):
    num += 1

    image = images.tolist()[0]
    label = labels.tolist()[0]

    if np.argmax(ai.start_work(image)) == np.argmax(np.array(label)):
        accuracy += 1

    if num == max_test_images:
        break

print(f"Точность: {round(accuracy / max_test_images *100, 1)}%")
print(f"Ошибка:   {round(100 - accuracy / max_test_images *100, 1)}% \n")

print("Время:", round((time() - start_time) / 60, 1), "мин")