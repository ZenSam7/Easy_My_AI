import Code_My_AI
from Games import Code_Snake


# Создаём Змейку
snake = Code_Snake.Snake(600, 400, 100, 2, max_num_steps=100, display_game=False)

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
ai.create_weights([9, 40, 40, 40, 4], add_bias_neuron=True)

ai.what_act_func = ai.kit_act_func.Sigmoid
ai.end_act_func = ai.kit_act_func.Softmax

actions = ["left", "right", "up", "down"]
ai.make_all_for_q_learning(actions, 0.3, 0.0, 0.1)


ai.name = "Snake_better"
ai.load()

ai.alpha = 5e-5
ai.batch_size = 1


learn_iteration = 0
while 1:
    learn_iteration += 1
    reward = 0

    if learn_iteration % 20_000 == 0:
        # Выводим максимальный и средний счёт змейки за 10_000 шагов
        max, mean = snake.get_max_mean_score()
        print(learn_iteration // 20_000, "\t\t", "Max:",max, "\t\t", "Mean:", round(mean, 1))

        ai.update()

    # ЗАПИСЫВАЕТ ДАННЫЕ В ОТВЕТ
    data = snake.get_blocks()

    action = ai.q_start_work(data)
    snake.step(action)

    # ОБУЧАЕМ
    ai.q_learning(data, reward, snake.get_future_state(action),
                  num_update_function=1, learning_method=1)
