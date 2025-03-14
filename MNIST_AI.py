from mnist import MNIST  # pip install mnist-py

import numpy as np
from easymyai import AI, AI_ensemble
from time import time

start_time = time()

mnist = MNIST()

ai = AI(
    architecture=[784, 100, 100, 10],
    add_bias=True,
    name="MNIST",
    alpha=2e-3,
    number_disabled_weights=0.0,
)

ai.main_act_func = ai.kit_act_func.tanh
ai.end_act_func = ai.kit_act_func.softmax

# ai.load()
ai.print_parameters()

ai.alpha = 3e-3
ai.disabled_neurons = 0.0

ai.impulse1 = 0.9
ai.impulse2 = 0.8
ai.l1 = 0e-5
ai.l2 = 0e-4


print("\nОбучение...")
for epoch in range(3):
    print(f"Эпоха #{epoch}")

    num, errors = 0, 0
    max_train_images, show_progress = 60_000, 0.10

    for images, labels in mnist.train_set.minibatches(batch_size=1):
        num += 1

        image = images.tolist()[0]
        label = labels.tolist()[0]
        ai.learning(image, label, use_Adam=True)

        if np.argmax(ai.predict(image)) != np.argmax(np.array(label)):
            errors += 1

        if num % int(max_train_images * show_progress) == 0:
            print(
                f">>> {int(num / max_train_images *100)}% \t\t",
                f"Images: {num} \t\t",
                f"Error: {round((errors / (max_train_images * show_progress)) *100, 1)}% \t\t",
                f"Summ weights: {int( sum([np.sum(np.abs(i)) for i in ai.weights]) )}",
            )
            # Для каждого вывода заново считаем ошибку
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

    if np.argmax(ai.predict(image)) == np.argmax(np.array(label)):
        accuracy += 1

    if num == max_test_images:
        break

print(f"Точность: {round(accuracy / max_test_images *100, 1)}%")
print(f"Ошибка:   {round(100 - accuracy / max_test_images *100, 1)}% \n")

print("Время:", round((time() - start_time) / 60, 1), "мин")
