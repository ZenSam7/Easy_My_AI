import Code_My_AI
from Snake import Snake

# Создаём Змейку
snake = Snake(800, 600, 100, 2, display_game=False)

def end():
    global reward
    reward = -3_000
def win():
    global reward
    reward = 1_000
snake.game_over_function = end
snake.eat_apple_function = win


# Создаём ИИ
ai = Code_My_AI.AI()
ai.create_weights([9, 15, 15, 4])

ai.end_activation_function = ai.activation_function.ReLU_2
ai.activation_function.value_range(-10, 10)


actions = ["left", "right", "up", "down"]
ai.make_all_for_q_learning(actions, 0.6, 0.0, 0.2)


# Обычная      ("Snake"):      9, 25, 25, 25, 4
# Маленькая    ("Snake_0.1"):  9, 15, 15, 4
version_snake = "Snake_0.1"

ai.load_data(version_snake)


learn_iteration = 0
num = 0
while 1:
    # Включаем режим отображения змейки для человека
    if snake.display_game == True:
        from time import sleep
        sleep(0.08)

    learn_iteration += 1
    reward = 0


    if learn_iteration % 10_000 == 0:
        # Выводим максимальный и средний счёт змейки за 10_000 шагов
        num += 1
        max, min, mean = snake.get_max_mean_score()
        print(f"#{num}", "Max Score:",max, "\t\t\t\t", "Mean Score:", round(mean, 1))
        snake.scores = []


        ai.delete_data(version_snake)
        ai.save_data(version_snake)

################# ЗАПИСЫВАЕТ ДАННЫЕ В ОТВЕТ

    data = snake.get_blocks()

    snake.step( ai.q_start_work(data) )

################# ОБУЧАЕМ

    ai.q_learning(data, reward, 1, 2.1, squared_error=False, recce_mode=False)
