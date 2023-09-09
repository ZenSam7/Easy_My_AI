import Code_My_AI
from Games import Code_Snake


# Создаём Змейку
snake = Code_Snake.Snake(600, 400, 100, 2, max_num_steps=50, display_game=True)

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
ai.create_weights([9, 35, 35, 35, 4], add_bias_neuron=True)

ai.what_act_func = ai.kit_act_funcs.ReLU_2

ai.alpha = 1e-5

actions = ["left", "right", "up", "down"]
ai.make_all_for_q_learning(actions, 0.4, 0.1, 0.15)


# Обычная                   ("Snake"):    9, 25, 25, 4
# Расширенная               ("Snake2"):   9, 35, 35, 35, 4
ai.name = "Snake2"
ai.load()

ai.load("Snake_original")
ai.save("Snake")


learn_iteration = 0
while 1:
    learn_iteration += 1
    reward = 0

    if learn_iteration == 5_000:
        # Выводим максимальный и средний счёт змейки за 10_000 шагов
        max, min, mean = snake.get_score()
        print("Max:", max, "\t\t" "Mean:", round(mean, 1))

        ai.delete()
        ai.save()

        learn_iteration = 0

    data = snake.get_blocks()

    action = ai.q_start_work(data)
    snake.step(action)


    ai.q_learning(data, reward, 1)