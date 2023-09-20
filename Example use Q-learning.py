import Code_My_AI
from Games import Game_for_Q_learning


game = Game_for_Q_learning.Game(7, 7)

# Состояния - нахождение в какой-либо клетке поля (координаты каждой клетки)
actions = ["left", "right", "up", "down"]


# Создаём ИИ
ai = Code_My_AI.AI()
ai.create_weights([2, 40, 40, 4], add_bias_neuron=True)

ai.what_act_func = ai.kit_act_func.Sigmoid
ai.end_act_func = ai.kit_act_func.Softmax

ai.batch_size = 1

ai.alpha = 5e-4

ai.make_all_for_q_learning(actions, ai.kit_upd_q_table.standart, 0.5, 0.05, 0.1)


reward, generation, num_win, number_steps = 0, 0, 0, 0

def died():
    global reward, generation, number_steps
    generation += 1
    reward = -100
    number_steps = 0  # Считаем сколько шагов, что бы не было "зацикливания" ИИ
def win():
    global reward, num_win, number_steps
    number_steps = 0
    reward = 1000
    num_win += 1
    print("WIN !", num_win, "\t", round(num_win/generation*100, 2))

game.game_over_function = died
game.win_function = win


# ai.load_data("Q")


learn_iteration = 0
while 1:
    number_steps += 1
    learn_iteration += 1

    if learn_iteration % 30 == 0:
        game.draw(generation)

        # Если слишком много шагов - убиваем
        if number_steps >= 100:
            game.game_over()


###################### ОТВЕТ ОТ НЕЙРОНКИ

    data = [[i for i in game.agent_coords]]# + game.walls_coords
    data = sum(data, [])

    where_move = ai.q_start_work(data)

    game.step(where_move)

###################### ОБУЧАЕМ

    ai.q_learning(data, reward, game.get_future_coords(where_move), learning_method=1)

    # Если не умерли и не победили, то 0 (т.е. штрафуем за лишние шаги)
    # (P.s. reward изменяется в game.win или game.game_over (в game.step),
    # и если они не сработали, то reward как был, так и остаётся 0)
    reward = -1
