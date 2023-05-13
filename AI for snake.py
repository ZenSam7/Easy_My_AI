import Code_My_AI
import Snake


# Загружаем последнее сохранение
# ai.load_data("Snake")


SNAKES = []
AIs = []
number_populations = 200       # Сколько змей в популяции (из скольких змей выбираем 2 наилучшие)


for _ in range(number_populations):
    ##### Создаём Змеек
    snake = Code_My_Snake.Snake(9, 9, 1, 11, game_over_function=None, display_game=False)

    def end():
        snake.generation += 1
    snake.game_over_function = end

    SNAKES.append(snake)


    ##### Создаём ИИ
    ai = Code_My_AI.AI()
    ai.create_weights([8, 10, 10, 4], add_bias_neuron=False)

    ai.what_activation_function = ai.activation_function.ReLU_2
    ai.activation_function.value_range(0, 1)
    ai.end_activation_function = ai.activation_function.Tanh

    ai.packet_size = 1
    ai.alpha = 1e-7

    AIs.append(ai)



iteration = 0   # Если змейка крутиться на месте, то iteration становиться очень большим, и для этой змеи GameOver()
while 1:
    iteration += 1

    for snake, ai in zip(SNAKES, AIs):    # Проводим по 1 итерации для каждой змейки поочереди

###################### ЗАПИСЫВАЕТ ДАННЫЕ В ОТВЕТ

        data = snake.get_range_to_blocks()

###################### ОТВЕТ ОТ НЕЙРОНКИ

        ai_answer = ai.start_work(data).tolist()
        if max(ai_answer) == ai_answer[0]:
            snake.step("left", snake.generation)
        elif max(ai_answer) == ai_answer[1]:
            snake.step("right", snake.generation)
        elif max(ai_answer) == ai_answer[2]:
            snake.step("up", snake.generation)
        elif max(ai_answer) == ai_answer[3]:
            snake.step("down", snake.generation)

###################### ОБУЧАЕМ

        if iteration >= 1_000:
            snake.game_over() # Что бы змея на месте не закручивалась

        if all([not i.alive for i in SNAKES]) and\
                number_populations >= 2:      # Если все мертвы И популяция >1
            # Среди всех змей выбираем 2, у которых счёт максимальный (среди остальных)
            # И скрещиваем эти 2 змеи (перемешиваем веса у нейронок этих змей)
            better_scores = sorted([snake.score for snake in SNAKES])[-2:]
            better_ais = []

            for i in range(len(SNAKES)):
                SNAKES[i].alive = True  # Всех воскрешаем

                if SNAKES[i].score in better_scores: # Если змея относится к 2м наилучшим
                    better_ais.append(AIs[i])    # Добавляем её нейронку к избранным

            better_ais[0].genetic_crossing_with(better_ais[1]) # Скрещиваем лучшего м лучшим

            for _ in range(len(AIs)): # Заменяем все старые ии на лучшего (с небольшими мутациями)
                better_ais[0].get_mutations(0.05)

            print(f"Поколение #{SNAKES[0].generation} | {better_scores[1]}")







