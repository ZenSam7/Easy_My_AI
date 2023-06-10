import Code_My_AI
from mnist import MNIST
import numpy as np

mnist = MNIST()


ai = Code_My_AI.AI()

ai.create_weights([784, 20, 20, 10], add_bias_neuron=False)

ai.what_act_func = ai.act_func.ReLU_2
ai.end_act_func  = ai.act_func.Sigmoid

ai.number_disabled_neurons = 0.0
ai.batch_size = 1
ai.alpha = 1e-6


name = "MNIST"
# ai.save_data(name)
ai.load_data(name)

# ai.print_how_many_parameters()



print("Обучение...")
num = 0
max_train_images, show_progress = 30_000, 0.05

for images, labels in mnist.train_set.minibatches(batch_size=int(max_train_images *show_progress)):
    num += int(max_train_images *show_progress)

    for index in range(int(max_train_images *show_progress)):
        image = images.tolist()[index]
        label = labels.tolist()[index]
        ai.learning(image, label, type_error=1, regularization=1, regularization_value=75)

    print(f">>> {int(num / max_train_images *100)}% \t\t Images: {num}")

    # Сохраняемся
    ai.delete_data(name)
    ai.save_data(name)

    if num >= max_train_images:
        break

print()


print("Тестирование")
accuracy = 0
max_test_images = 700

for images, labels in mnist.test_set.minibatches(batch_size=max_test_images):
    for index in range(max_test_images):
        image = images.tolist()[index]
        label = labels.tolist()[index]

        if np.argmax(ai.start_work(image)) == np.argmax(np.array(label)):
            accuracy += 1

    print(f"Точность: {round(accuracy / max_test_images *100, 1)}%")
    print(f"Ошибка:   {round(100 - accuracy / max_test_images *100, 1)}%")
    break
