from My_AI import AI_ensemble, AI
from Games import Code_Snake
from time import time

start = time()

# Создаём Змейку
snake = Code_Snake.Snake(700, 500, 3, 0, max_steps=50, display_game=False,
                         dead_reward=-100, win_reward=200)

# Создаём ИИ
ai = AI(architecture=[9, 40, 40, 40, 4],
                 add_bias_neuron=True, name="Snake")

ai.what_act_func = ai.kit_act_func.tanh
ai.end_act_func = ai.kit_act_func.tanh

ai.make_all_for_q_learning(("left", "right", "up", "down"),
                           ai.kit_upd_q_table.future,
                           0.5, 0.0, 0.1)

# ai.load()
ai.print_parameters()

ai.alpha = 1e-4
ai.impulse = 0.7


learn_iteration = 0
while 1:
    learn_iteration += 1
    reward = 0

    if learn_iteration % 20_000 == 0:
        # Выводим максимальный и средний счёт змейки за 20_000 шагов
        max, mean = snake.get_max_mean_score()
        print(learn_iteration // 20_000, "\t\t",
              "Max:", max, "\t\t",
              "Mean:", round(mean, 1), "\t\t",
              int(time() - start), "s", "\t\t",
              "Amount States:", len(ai.q_table.keys()))
        start = time()

        # print(ai._momentums[0][0, :6], ai._velocities[0][0, :6])
        ai.update(check_ai=True)

    # Записываем данные в ответ
    data = snake.get_blocks(3)

    action = ai.q_predict(data)
    reward = snake.step(action)

    # Обучаем
    ai.q_learning(data, reward, learning_method=1, squared_error=True,)
