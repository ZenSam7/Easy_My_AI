from My_AI import AI_ensemble, AI
from Games import Code_Snake
from time import time

start = time()


# Создаём Змейку
snake = Code_Snake.Snake(700, 600, 4, 0, display_game=False,
                         dead_reward=-100, win_reward=100)

# Создаём ИИ
ai = AI(architecture=[9, 200, 200, 4],
                      add_bias_neuron=True, name="Snake")

ai.what_act_func = ai.kit_act_func.tanh
ai.end_act_func = ai.kit_act_func.tanh  # Только tanh!

ai.make_all_for_q_learning(("left", "right", "up", "down"),
                           ai.kit_upd_q_table.standart,
                           0.4, 0.05, 0.1)

# ai.load()
ai.print_parameters()

ai.alpha = 1e-4
ai.batch_size = 1

ai.epsilon = 0.05

# # Используем уже обученную глобальную Q-таблицу  (чтобы ИИшки учились
# # молниеносно, и можно было сразу видеть влияние каких-то факторов на ИИшку)
# q_table_ai = AI(name="q_table_ai")
# q_table_ai.load("q_table_ai")
# ai.q_table = q_table_ai.q_table


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
              "Len States:", len(ai.q_table.keys()))
        start = time()

        ai.update(check_ai=True)

        # # Обновляем и глобальную Q-таблицу
        # q_table_ai.q_table = ai.q_table
        # q_table_ai.update()

    # Записываем данные в ответ
    data = snake.get_blocks(3)

    action = ai.q_predict(data)
    reward = snake.step(action)

    # Обучаем
    ai.q_learning(data, reward, learning_method=1, squared_error=True)
