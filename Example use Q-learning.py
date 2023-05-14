import Code_My_AI
import Game_for_Q_learning


game = Game_for_Q_learning.Game(6, 5)

# Состояния - нахождение в какой-либо клетке поля (координаты каждой клетки)
actions = ["left", "right", "up", "down"]


# Создаём ИИ
ai = Code_My_AI.AI()
ai.create_weights([2, 10, 10, 4], add_bias_neuron=True)

ai.what_activation_function = ai.activation_function.ReLU_2
ai.activation_function.value_range(0, 1)
ai.end_activation_function = ai.activation_function.ReLU_2

ai.alpha = 1e-3
ai.q_alpha = 1e-2

ai.make_all_for_q_learning(actions, 0.7, 0.2)



reward, generation, num_win = 0, 0, 0

def died():
    global reward, generation
    generation += 1
    reward = -100
def win():
    global reward, generation, num_win
    generation += 1
    reward = 10_000
    num_win += 1
    print("WIN !", num_win, " ", round(num_win/generation*100, 2))

game.game_over_function = died
game.win_function = win



learn_iteration = 0
while 1:
    learn_iteration += 1

    if learn_iteration % 50 == 0:
        game.draw(generation)

###################### ОТВЕТ ОТ НЕЙРОНКИ

    data = [i +0.1 for i in game.agent_coords]   # Нельзя чтобы на входе были 0

    where_move = ai.q_start_work(data)

    game.step(where_move)

###################### ОБУЧАЕМ

    ai.q_learning(data, where_move, reward, game.get_future_coords(where_move), 1)   # Лучше всего выбрать функцию 3


    # Если не умерли и не победили, то 0 (т.е. штрафуем за лишние шаги)
    # (P.s. reward изменяется в game.win или game.game_over (в game.step), и если они не сработали, то reward как был, так и остаётся 0)
    reward = -1


