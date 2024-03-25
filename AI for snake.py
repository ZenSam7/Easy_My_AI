from easymyai import AI_ensemble, AI
from Games import Snake
from time import time

# Создаём Змейку
snake = Snake(9, 5, amount_food=1, amount_walls=0,
              max_steps=100, display_game=True, cell_size=120)

# Загружаем лучшую нейронку
ai = AI_ensemble(1)
ai.load("")
ai.print_parameters()


start_time, learn_iteration = time(), 0
while 1:
    learn_iteration += 1

    if learn_iteration % 50_000 == 0:
        # Выводим максимальный и средний счёт змейки за 50_000 шагов
        max, mean = snake.get_max_mean_score()
        print(
            str(learn_iteration//1000)+"_000",
            "\t\tMax:", max,
            "\t\tMean:", round(mean, 1),
            "\t\t", int(time() - start_time), "s",
            "\t\tAmount States:", len(ai.q_table.keys()),
        )
        start_time = time()

    # Записываем данные которые видит Змейка
    data = snake.get_blocks(3)
    # data = snake.get_ranges_to_blocks()

    action = ai.q_predict(data)
    snake.step(action)
