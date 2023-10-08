from Code_My_AI import AI
from Games import Code_Snake
from time import time
from profilehooks import profile

start = time()

def end():
    global reward
    reward = -10
def win():
    global reward
    reward = 50

# Создаём Змейку
snake = Code_Snake.Snake(600, 500, 2, 2, max_num_steps=100, display_game=True,
                         game_over_function=end, eat_apple_function=win)

# Создаём ИИ
ai = AI(architecture=[9, 100, 100, 100, 4], add_bias_neuron=True, name="Snake")

ai.what_act_func = ai.kit_act_func.tanh
ai.end_act_func = ai.kit_act_func.softmax

actions = ("left", "right", "up", "down")
ai.make_all_for_q_learning(actions, ai.kit_upd_q_table.standart, 0.1, 0.02, 0.1)

ai.load()
ai.print_how_many_parameters()

ai.alpha = 1e-5
ai.batch_size = 10

ai.epsilon = 0.0


learn_iteration = 0
while 1:
    learn_iteration += 1
    reward = 0

    if learn_iteration % 30_000 == 0:
        # Выводим максимальный и средний счёт змейки за 30_000 шагов
        max, mean = snake.get_max_mean_score()
        print(learn_iteration // 30_000, "\t\t",
              "Max:", max, "\t\t",
              "Mean:", round(mean, 1), "\t\t",
              int(time() - start), "s", "\t\t",
              "Len States:", len(ai.q_table.keys()))
        start = time()

        ai.update()

    # Записываем данные в ответ
    data = snake.get_blocks(3)

    action = ai.q_start_work(data)
    snake.step(action)

    # Обучаем
    ai.q_learning(data, reward, snake.get_future_state(action),
                  recce_mode=False, learning_method=1)
