import Code_My_AI
from Games import Game_for_Q_learning

game = Game_for_Q_learning.Game(7, 4)

# Состояния - нахождение в какой-либо клетке поля (координаты каждой клетки)
actions = ["left", "right", "up", "down"]


# Создаём ИИ
ai = Code_My_AI.AI()
ai.create_weights([2, 10, 4], add_bias_neuron=False)

ai.what_act_func = ai.act_func.ReLU_2
ai.act_func.value_range(-10, 10)
ai.end_act_func = ai.act_func.ReLU_2

ai.alpha = 1e-4

ai.make_all_for_q_learning(actions, 0.4, 0.0, 0.1)


reward, generation, num_win, number_steps = 0, 0, 0, 0


def died():
    global reward, generation, number_steps
    generation += 1
    reward = -100
    number_steps = 0  # Считаем сколько шагов, что бы не было "зацикливания" ИИ


def win():
    global reward, num_win, number_steps
    number_steps = 0
    reward = 10_000
    num_win += 1
    print("WIN !", num_win, "\t", round(num_win / generation * 100, 2))


game.game_over_function = died
game.win_function = win


# ai.save_data("Q")


learn_iteration = 0
while True:
    number_steps += 1
    learn_iteration += 1

    if learn_iteration % 20 == 0:
        game.draw(generation)

        # Если слишком много шагов - убиваем
        if number_steps >= 200:
            game.game_over()

    # if learn_iteration % 5_000 == 0:
    #     ai.delete_data("Q")
    #     ai.save_data("Q")


# ОТВЕТ ОТ НЕЙРОНКИ

    data = [[i + 0.01 for i in game.agent_coords]]  # + game.walls_coords
    data = sum(data, [])

    game.step(ai.q_start_work(data))

# ОБУЧАЕМ

    ai.q_learning(data, reward, 2, 2.4, type_error=1, recce_mode=False)

    # Если не умерли и не победили, то 0 (т.е. штрафуем за лишние шаги)
    # (P.s. reward изменяется в game.win или game.game_over (в game.step), и если они не сработали, то reward как был, так и остаётся 0)
    reward = -2
