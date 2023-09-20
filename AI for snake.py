import Code_My_AI
from Games import Code_Snake
import os
from profilehooks import profile
from time import time

start = time()

# Создаём Змейку
snake = Code_Snake.Snake(700, 500, 100, 3, max_num_steps=100, display_game=False)

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
ai.create_weights([9, 100, 100, 4], add_bias_neuron=True)
ai.name = "Snake_2"

ai.what_act_func = ai.kit_act_func.Tanh
ai.end_act_func = ai.kit_act_func.Softmax

actions = ["left", "right", "up", "down"]
ai.make_all_for_q_learning(actions, ai.kit_upd_q_table.standart, 0.4, 0.0, 0.1)

# ai.load()
ai.print_how_many_parameters()

ai.alpha = 1e-4
ai.batch_size = 1

# TODO: Научиться сохранять func_upd_q_table

learn_iteration = 0
while 1:
    learn_iteration += 1
    reward = 0

    if learn_iteration % 30_000 == 0:
        # Выводим максимальный и средний счёт змейки за 10_000 шагов
        max, mean = snake.get_max_mean_score()
        print(learn_iteration // 30_000, "\t\t",
              "Max:", max, "\t\t",
              "Mean:", round(mean, 1), "\t\t",
              int(time() - start), " s", "\t\t",
              "Len States:", len(ai.states))
        start = time()

        ai.update()

    # ЗАПИСЫВАЕТ ДАННЫЕ В ОТВЕТ
    data = snake.get_blocks(3)

    action = ai.q_start_work(data)
    snake.step(action)

    # ОБУЧАЕМ
    ai.q_learning(data, reward, snake.get_future_state(action),
                  recce_mode=True, update_q_table=True, add_new_states=True,
                  learning_method=1)
