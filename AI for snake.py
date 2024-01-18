from easymyai import AI_ensemble, AI
from Games import Snake
from time import time

start = time()

# Создаём Змейку
snake = Snake(7, 5, 2, 0,
              max_steps=40, display_game=False,
              dead_reward=-100, win_reward=200, cell_size=120)

# Создаём ИИ
ai = AI_ensemble(3, architecture=[25, 100, 100, 100, 100, 4], add_bias_neuron=True,
                 name="Snake_25")

ai.what_act_func = ai.kit_act_func.tanh
ai.end_act_func = ai.kit_act_func.softmax

ai.make_all_for_q_learning(("left", "right", "up", "down"),
                           ai.kit_upd_q_table.standart,
                           0.7, 0.1, 0.1)

# ai.load()
ai.print_parameters()

ai.alpha = 1e-3

ai.impulse1 = 0.75
ai.impulse2 = 0.9
ai.l1 = 0.0
ai.l2 = 0.0


learn_iteration: int = 0
while 1:
    learn_iteration += 1
    reward = 0

    if learn_iteration % 50_000 == 0:
        # Выводим максимальный и средний счёт змейки за 50_000 шагов
        max, mean = snake.get_max_mean_score()
        print(
            learn_iteration // 50_000,
            "\t\tMax:", max,
            "\t\tMean:", round(mean, 1),
            "\t\t", int(time() - start), "s",
            "\t\tAmount States:", len(ai.q_table.keys()),
        )
        start = time()

        ai.epsilon = ai.epsilon * 0.85

        ai.update(check_ai=True)

    # Записываем данные в ответ
    data = snake.get_blocks(5)

    action = ai.q_predict(data)
    reward = snake.step(action)

    # Обучаем
    ai.q_learning(data, reward, learning_method=1, squared_error=False, use_Adam=True)
