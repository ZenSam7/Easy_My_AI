import Code_My_AI
from Snake import Snake


# Создаём Змейку
snake = Snake(700, 500, 100, 3, display_game=False)

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
ai.create_weights([8, 20, 20, 20, 4], add_bias_neuron=False)

ai.what_activation_function = ai.activation_function.ReLU_2
ai.activation_function.value_range(-10, 10)
ai.end_activation_function = ai.activation_function.ReLU_2

ai.alpha = 1e-7

actions = ["left", "right", "up", "down"]
ai.make_all_for_q_learning(actions, 0.1, 0.0, 1)



ai.load_data("Snake")


learn_iteration = 0
while 1:
    learn_iteration += 1
    reward = 0

    if learn_iteration % 10_000 == 0:
        print(snake.max_score)
        snake.max_score = 0
        ai.delete_data("Snake")
        ai.save_data("Snake")

################# ЗАПИСЫВАЕТ ДАННЫЕ В ОТВЕТ

    data = snake.get_blocks()

    action = ai.q_start_work(data)
    snake.step(action, snake.generation)

################# ОБУЧАЕМ

    ai.q_learning(data, reward, action, 1)

