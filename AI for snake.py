import Code_My_AI
from Snake import Snake


len_population = 10     # >2
how_many_AI_cross = 1   # <len_population  &  ЧЁТНОЕ!!


SNAKES = []
AIs = []
for _ in range(len_population):
    # Создаём Змейку
    snake = Snake(500, 400, 100, 1)

    def end():
        global reward
        reward = -50
    def win():
        global reward
        reward = 100
    snake.game_over_function = end
    snake.eat_apple_function = win

    SNAKES.append(snake)


    # Создаём ИИ
    ai = Code_My_AI.AI()
    ai.create_weights([9, 20, 20, 4])

    ai.what_act_func = ai.act_func.Tanh
    ai.end_act_func  = ai.act_func.Tanh
    ai.act_func.value_range(0, 1)

    actions = ["left", "right", "up", "down"]
    ai.make_all_for_q_learning(actions, gamma=0.1, epsilon=0.0, q_alpha=0.01)

    ai.batch_size = 1
    ai.alpha = 1e-4
    ai.number_disabled_weights = 0.1

    AIs.append(ai)


# Загружаем всех
for i in range(len(AIs)):
    # Каждой ИИ свой место и свой имя
    version_snake = "Snake_0.1~" + str( i )
    AIs[i].load_data(version_snake)


learn_iteration, num = 0, 0
while 1:
    for ai, snake in zip(AIs, SNAKES):
        learn_iteration += 1
        reward = 0


        # Выводим максимальный и средний счёт каждой змейки за 1_000 шагов
        if learn_iteration % 2_000 == 0:
            num += 1
            for i in range(len_population):
                max, min, mean = SNAKES[i].get_score()
                print(f"#{num}.{i}",  "\tMax Score:",max,  "\t\t\t\t Mean Score:", round(mean, 1))
            print()


            # Каждые 20 шагов скрещиваем 2 змеи и добавляем мутации, а потом продолжаем обучать Q-обучением
            # (которое корректирует веса обратным распространением) ((Т.е. 3 вида обучения в 1 проекте получается))
            if num % 20 == 0:
                # Выбираем 2 лучших змейки
                SCORES = []
                for snake in SNAKES:
                    _, _, mean = snake.get_score()
                    SCORES.append(mean)

                SCORES.sort()
                SCORES = SCORES[-1*how_many_AI_cross:]  # Выбираем how_many_AI_cross наилучших
                best_ais = []

                for i in range(len_population):
                    _, _, mean = SNAKES[i].get_score()
                    SNAKES[i].scores = [0]  # Очищаем
                    if mean in SCORES:      # Если ИИ совпадает с лучшими, то добавляем
                        best_ais.append( AIs[i] )


                # # Скрещиваем 1 с 2, 3 с 4 ...  Пока не останется одна, со всеми равномерно скрещенная, ИИ
                # while len(best_ais) != 1:
                #     for i in range(0, len(best_ais) //2):
                #         best_ais[i].genetic_crossing_with(best_ais[i +1])
                #         crossed_ai = best_ais[i]
                #
                #         best_ais.pop(i)
                #         best_ais.pop(i)
                #
                #         best_ais.insert(i, crossed_ai)

                best_ai = best_ais[0]   # Просто вытаскиваем из списка
                # Создаём клонов (с мутациями)
                from copy import deepcopy
                AIs.clear()
                for _ in range(len_population):
                    best_ai.get_mutations(0.02)
                    AIs.append( deepcopy(best_ai) )

                print("Mutating", end="\n\n")


                # И сохраняемся
                for i in range(len_population):
                    version_snake = "Snake_0.1~" + str(i)
                    AIs[i].delete_data(version_snake)
                    AIs[i].save_data(version_snake)


    ################# ОБУЧАЕМ

        data = snake.get_blocks()

        snake.step( ai.q_start_work(data) )

        ai.q_learning(data, reward, num_update_function=1, learning_method=2.3, type_error=1, recce_mode=False)
