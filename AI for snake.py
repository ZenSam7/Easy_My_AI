from easymyai import AI_ensemble, AI
from Learning_Snakes import snake_parameters, ais_parameters
from Games import Snake

# Создаём Змейку
snake = Snake(snake_parameters["wight"], snake_parameters["height"],
              amount_food=snake_parameters["amount_food"], max_steps=snake_parameters["max_steps"],
              display_game=True, cell_size=120)

# Загружаем лучшую нейронку
ai = AI_ensemble(1)
ai.load("Snake_12.4")
ai.print_parameters()


learn_iteration = 0
while 1:
    learn_iteration += 1

    if learn_iteration % ais_parameters["max_learn_iteration"] == 0:
        # Выводим максимальный и средний счёт змейки за 50_000 шагов
        _, mean = snake.get_max_mean_score()
        print(
            str(learn_iteration//ais_parameters["max_learn_iteration"]),
            "\t\tMean:", round(mean, 1),
            "\t\tAmount States:", len(ai.q_table.keys()),
        )

    # Записываем данные которые видит Змейка
    data = snake.get_blocks(ais_parameters["visibility_range"])
    # data = snake.get_ranges_to_blocks()

    action = ai.q_predict(data)
    snake.step(action)
