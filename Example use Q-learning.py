from My_AI import AI, AI_ensemble
from Games import Q_Game

game = Q_Game(7, 7)

# Создаём ИИ
ai = AI()
ai.create_weights([2, 50, 50, 4], add_bias=True)
# Состояния - нахождение в какой-либо клетке поля (координаты каждой клетки)
ai.make_all_for_q_learning(("left", "right", "up", "down"),
                           ai.kit_upd_q_table.standart, 0.5, 0.05, 0.1)

ai.what_act_func = ai.kit_act_func.tanh
ai.end_act_func = ai.kit_act_func.tanh

ai.batch_size = 1
ai.alpha = 1e-2


reward, generation, num_win, number_steps = 0, 0, 0, 0

def died():
    global reward, generation, number_steps
    generation += 1
    reward = -10
    number_steps = 0  # Считаем сколько шагов, что бы не было "зацикливания" ИИ
def win():
    global reward, num_win, number_steps
    number_steps = 0
    reward = 1000
    num_win += 1
    print("WIN !", num_win, "\t", round(num_win/generation, 4))

game.game_over_function = died
game.win_function = win


# ai.load_data("Q")


learn_iteration = 0
while 1:
    number_steps += 1
    learn_iteration += 1
    reward = 0

    if learn_iteration % 30 == 0:
        game.draw(generation)

        # Если слишком много шагов - убиваем
        if number_steps >= 100:
            game.game_over()


    # Ответ от ИИшки

    data = [game.agent_coords] # + game.walls_coords
    data = sum(data, [])

    where_move = ai.q_predict(data)

    game.step(where_move)

    # Обучаем

    ai.q_learning(data, reward, learning_method=1, squared_error=True)
