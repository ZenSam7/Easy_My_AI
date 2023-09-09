import Code_My_AI
from Games import Code_Snake
import numpy as np


# Создаём Змейку
snake = Code_Snake.Snake(600, 400, 100, 3, max_num_steps=50, display_game=False)

def end():
    global reward
    reward = -100
    snake.generation += 1
def win():
    global reward
    reward = 1_000
snake.game_over_function = end
snake.eat_apple_function = win


# Создаём ИИ
ai = Code_My_AI.AI()
ai.create_weights([9, 20, 20, 4], add_bias_neuron=True)

ai.what_act_func = ai.kit_act_funcs.Tanh

ai.alpha = 1e-5

actions = ["left", "right", "up", "down"]
ai.make_all_for_q_learning(actions, 0.3, 0.1, 0.1)


# Обычная         (обучена) ("Snake"):    9, 25, 25, 4
ai.name = "Snake_test"
ai.load()

# ai.load("Snake_original")
# ai.save("Snake")


learn_iteration = 0
while 1:
    learn_iteration += 1
    reward = 0

    if learn_iteration % 20_000 == 0:
        # Выводим максимальный и средний счёт змейки за 10_000 шагов
        max, _, mean = snake.get_score()
        print(learn_iteration//10_000, "\t",
              "Max:", max, "\t\t",
              "Mean:", round(mean, 1), "\t\t",
              f"Summ weights: {int( sum([np.sum(np.abs(i)) for i in ai.weights]) )}")

        ai.update()

    data = snake.get_blocks()

    action = ai.q_start_work(data)
    snake.step(action)


    ai.q_learning(data, reward, 1)