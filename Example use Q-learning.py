import Code_My_AI
import Game_for_Q_learning


game = Game_for_Q_learning.Game(5, 4)

# Состояния - нахождение в какой-либо клетке поля (координаты каждой клетки)
actions = ["left", "right", "up", "down"]


# Создаём ИИ
ai = Code_My_AI.AI()
ai.create_weights([2, 13, 4], add_bias_neuron=True)

ai.what_activation_function = ai.activation_function.ReLU
ai.activation_function.value_range(0, 10)
ai.end_activation_function = ai.activation_function.ReLU_2

ai.packet_size = 1

ai.alpha = 1e-4



reward, generation, num_win = 0, 0, 0

def died():
    global reward, generation
    generation += 1
    reward = -100
def win():
    global reward, generation, num_win
    generation += 1
    reward = 1000
    num_win += 1
    print("WIN !", num_win)

game.game_over_function = died
game.win_function = win


ai.make_all_for_q_learning(actions, 2.2, 0.15)



learn_iteration = 0
while 1:
    learn_iteration += 1

    if learn_iteration % 30 == 0:
        game.draw(generation)

###################### ОТВЕТ ОТ НЕЙРОНКИ

    data = [i +0.5 for i in game.agent_coords]
    ai_answer = ai.start_work(data).tolist()

    where_move = ""
    if max(ai_answer) == ai_answer[0]:
        where_move = "left"
    elif max(ai_answer) == ai_answer[1]:
        where_move = "right"
    elif max(ai_answer) == ai_answer[2]:
        where_move = "up"
    elif max(ai_answer) == ai_answer[3]:
        where_move = "down"

    game.step(where_move)

###################### ОБУЧАЕМ

    ai.q_learning([i +0.5 for i in game.agent_coords], # Нельзя чтобы на входе были 0
                  where_move, reward, game.get_future_coords(where_move))


    # Если не умерли и не победили, то -0.1 (т.е. штрафуем за лишние шаги)
    # (P.s. reward изменяется в game.win* или game.game_over* (в game.step), и если они не сработали, то reward как был, так и остаётся -0.1)
    # *точнее в функциях, которые мы им передаём (win и died)
    reward = -1


