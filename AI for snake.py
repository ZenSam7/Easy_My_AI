from easymyai import AI_ensemble, AI
from Games import Snake
from time import time
from random import random

start = time()

# Создаём Змейку
snake = Snake(7, 5, amount_food=1, amount_walls=0,
              max_steps=100, display_game=False,
              dead_reward=-400, win_reward=200, cell_size=120)

# Создаём автоэнкодер
decoder = AI(architecture=[9, 20, 10, 9],  # Выход потом обрежем
             add_bias_neuron=True, name="snake_decoder")
encoder = AI(architecture=[4, 50, 20, 4],  # Вход потом обрежем
             add_bias=True, name="snake_encoder")

# Обучаем энкодер и декодер
for __ in range(2000):
    data_decoder = [1 - 2*random() for _ in range(9)]
    data_encoder = [random() for _ in range(4)]

    for ___ in range(100):
        decoder.learning(data_decoder, data_decoder, use_Adam=False)
        encoder.learning(data_encoder, data_encoder, use_Adam=False)
print("Автоэнкодер обучен!")


# Если сделать тут на входе большое число, то будет слишком много
# вариаций состояний и обучения не будет
mind = AI(architecture=[10, 100, 100, 50],
          add_bias=True, name="snake_mind")

# Да, мы оставляем softmax
mind.end_act_func = mind.kit_act_func.softmax

encoder.make_all_for_q_learning(("left", "right", "up", "down"))
mind.make_all_for_q_learning([str(i) for i in range(50)],
                             encoder.kit_upd_q_table.simple,
                             gamma=.6, epsilon=.0, q_alpha=.1)

# Обрезаем автоэнкодеры
encoder.weights, encoder.biases = encoder.weights[1:], encoder.biases[1:]
decoder.weights, decoder.biases = decoder.weights[:-1], decoder.biases[:-1]

# Энкодеры не обучаем
mind.print_parameters()

mind.alpha = 1e-3

mind.impulse1 = 0.7
mind.impulse2 = 0.9

learn_iteration: int = 0
while 1:
    learn_iteration += 1
    reward = 0

    if learn_iteration % 50_000 == 0:
        # Выводим максимальный и средний счёт змейки за 50_000 шагов
        max, mean = snake.get_max_mean_score()
        print(
            str(learn_iteration//1000)+"_000",
            "\t\tMax:", max,
            "\t\tMean:", round(mean, 1),
            "\t\t", int(time() - start), "s",
            "\t\tAmount States:", len(mind.q_table.keys()),
        )
        start = time()
        mind.update(check_ai=False)

    data = snake.get_blocks(3)

    decoder_ans = decoder.predict(data).tolist()[0]

    action = encoder.q_predict(mind.predict(decoder_ans).tolist()[0])

    reward = snake.step(action)

    # Обучаем только основную часть
    mind.q_learning(decoder_ans, reward, learning_method=2.1,
                    squared_error=False, use_Adam=True, rounding=1)
