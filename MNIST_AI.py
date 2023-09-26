import Code_My_AI
from mnist import MNIST
import numpy as np
from time import time

start_time = time()

mnist = MNIST()

ai = Code_My_AI.AI()

ai.create_weights([784, 50, 50, 50, 10], add_bias_neuron=True)
ai.name = "MNIST"

ai.what_act_func = ai.kit_act_func.Tanh
ai.end_act_func = ai.kit_act_func.Softmax

ai.number_disabled_weights = 0.0

# ai.load()
ai.print_how_many_parameters()

ai.batch_size = 1
ai.alpha = 1e-4



print("\nОбучение...")

for cycle in range(1):
    print(f"Эпоха #{cycle}")

    num, errors = 0, 0
    max_train_images, show_progress = 60_000, 0.10

    for images, labels in mnist.train_set.minibatches(batch_size=1):
        num += 1

        image = images.tolist()[0]
        label = labels.tolist()[0]
        ai.learning(
            image,
            label,
            # impulse_coefficient=0.9,
        )

        if np.argmax(ai.start_work(image)) != np.argmax(np.array(label)):
            errors += 1

        if num % int(max_train_images * show_progress) == 0:
            print(
                f">>> {int(num / max_train_images *100)}% \t\t",
                f"Images: {num} \t\t",
                f"Error: {round((errors / (max_train_images * show_progress)) *100, 1)}% \t\t",
                f"Summ weights: {int( sum([np.sum(np.abs(i)) for i in ai.weights]) )}",
            )

            errors = 0

            # Сохраняемся
            ai.update()

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
