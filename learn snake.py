from easymyai import AI_ensemble, AI
from Games import Snake
from time import time

start = time()

# Создаём Змейку
snake = Snake(7, 5, amount_food=1, amount_walls=0,
              max_steps=100, display_game=False,
              dead_reward=-20, win_reward=10, cell_size=120)

# Создаём ансамбль ИИ
ai = AI_ensemble(3, architecture=[9, 100, 100, 100, 4],
                 add_bias_neuron=True, name="Test_snake")

ai.end_act_func = ai.kit_act_func.softmax

ai.make_all_for_q_learning(("left", "right", "up", "down"),
                           ai.kit_upd_q_table.future,
                           gamma=.8, epsilon=.0, q_alpha=.1)

ai.load()
ai.print_parameters()

ai.alpha = 1e-3

ai.impulse1 = 0.6
ai.impulse2 = 0.9


learn_iteration: int = 0
while 1:
    learn_iteration += 1
    reward = 0

    if learn_iteration % 20_000 == 0:
        # Выводим максимальный и средний счёт змейки за 50_000 шагов
        _, mean = snake.get_max_mean_score()
        print(
            str(learn_iteration//20_000),
            "\t\tMean:", round(mean, 1),
            "\t\t", int(time() - start), "s",
            "\t\tAmount States:", len(ai.q_table.keys()),
        )
        start = time()
        ai.update(check_ai=True)

        if mean > 17:
            print("ЛУЧШАЯ ЗМЕЙКА!!!!!!")
            exit()

    # Записываем данные которые видит Змейка
    # (можно использовать один из 2х вариантов:)
    data = snake.get_blocks(3)
    # data = snake.get_ranges_to_blocks()

    action = ai.q_predict(data)
    reward = snake.step(action)

    # Обучаем
    ai.q_learning(data, reward, learning_method=1,
                  squared_error=False, use_Adam=True)
