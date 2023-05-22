import Code_My_AI
from Snake import Snake


# Создаём Змейку
snake = Snake(600, 500, 100, 3, display_game=False)

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
ai.create_weights([9, 35, 35, 35, 4], add_bias_neuron=False)

ai.what_activation_function = ai.activation_function.ReLU_2
ai.activation_function.value_range(-10, 10)
ai.end_activation_function = ai.activation_function.ReLU_2

ai.alpha = 1e-3

actions = ["left", "right", "up", "down"]
ai.make_all_for_q_learning(actions, 0.3, 0.1, 0.1)



# Обычная                   ("Snake"):    9, 25, 25, 4
# Расширенная (необучена)   ("Snake2"):   9, 35, 35, 35, 4
version_snake = "Snake2"

ai.load_data(version_snake)

from time import sleep
learn_iteration = 0
while 1:
    sleep(0.07)
    learn_iteration += 1
    reward = 0

    if learn_iteration % 10_000 == 0:
        # Выводим максимальный и средний счёт змейки за 10_000 шагов
        max, mean = snake.get_max_mean_score()
        print("Max:",max, "\t\t\t\t", "Mean:", round(mean, 1))
        snake.scores = []

        ai.delete_data(version_snake)
        ai.save_data(version_snake)

################# ЗАПИСЫВАЕТ ДАННЫЕ В ОТВЕТ

    data = snake.get_blocks()

    action = ai.q_start_work(data)
    snake.step(action, snake.generation)

################# ОБУЧАЕМ

    ai.q_learning(data, reward, snake.get_future_state(action), 1)
