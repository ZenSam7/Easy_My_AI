import Code_My_AI
import Game_for_Q_learning


game = Game_for_Q_learning.Game(7, 4)

# Состояния - нахождение в какой-либо клетке поля (координаты каждой клетки)
actions = ["left", "right", "up", "down"]


# Создаём ИИ
ai = Code_My_AI.AI()
ai.create_weights([2, 20, 4], add_bias_neuron=True)

ai.what_activation_function = ai.activation_function.ReLU_2
ai.activation_function.value_range(-10, 10)
ai.end_activation_function = ai.activation_function.ReLU_2

ai.alpha = 1e-5

ai.make_all_for_q_learning(actions, 0.4, 0.02, 0.1)



reward, generation, num_win, number_steps = 0, 0, 0, 0

def died():
    global reward, generation, number_steps
    generation += 1
    reward = -100
    number_steps = 0  # Считаем сколько шагов, что бы не было "зацикливания" ИИ
def win():
    global reward, num_win, number_steps
    number_steps = 0
    reward = 2000
    num_win += 1
    print("WIN !", num_win, "\t", round(num_win/generation*100, 2))

game.game_over_function = died
game.win_function = win


# ai.load_data("Q")


learn_iteration = 0
while 1:
    number_steps += 1
    learn_iteration += 1

    if learn_iteration % 25 == 0:
        game.draw(generation)

        # Если слишком много шагов - убиваем
        if number_steps >= 100:
            game.game_over()


    # if learn_iteration % 5_000 == 0:
    #     ai.delete_data("Q")
    #     ai.save_data("Q")


###################### ОТВЕТ ОТ НЕЙРОНКИ

    data = [[i +0.01 for i in game.agent_coords]]# + game.walls_coords
    data = sum(data, [])

    game.step( ai.q_start_work(data) )

###################### ОБУЧАЕМ

    ai.q_learning(data, reward, 2, 2.1, recce_mode=True)   # Лучше всего выбрать функцию 2


    # Если не умерли и не победили, то 0 (т.е. штрафуем за лишние шаги)
    # (P.s. reward изменяется в game.win или game.game_over (в game.step), и если они не сработали, то reward как был, так и остаётся 0)
    reward = -2
