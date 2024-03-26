from easymyai import AI_ensemble
from Games import Snake
from multiprocessing import Process, Queue
from multiprocessing.managers import BaseManager
from time import time, sleep

start_time = time()

# Параметры, которые мы вставим во все скрипты
snake_parameters = {
    "wight": 9,
    "height": 7,
    "max_steps": 100,

    "dead_reward": -10,
    "win_reward":  20,

    "amount_food": 1,
}

ais_parameters = {
    "amount_ais": 3,
    "architecture": [9, 100, 100, 100, 4],
    "visibility_range": 3,

    "alpha": 1e-3,
    "impulse1": 0.7,
    "impulse2": 0.9,

    "gamma": .6,
    "epsilon": .01,
    "q_alpha": .1,

    "func_update_q_table": AI_ensemble(1).kit_upd_q_table.future,
    "max_learn_iteration": 50_000,
    "learning_method": 1,
    "squared_error": False,
    "use_Adam": False,
}


def script_learns(snake_parameters, ais_parameters, found_best_snake):
    # Создаём Змейку
    snake = Snake(snake_parameters["wight"], snake_parameters["height"],
                  amount_food=snake_parameters["amount_food"], max_steps=snake_parameters["max_steps"],
                  dead_reward=snake_parameters["dead_reward"], win_reward=snake_parameters["win_reward"])

    # Создаём ансамбль ИИ
    ai = AI_ensemble(ais_parameters["amount_ais"],
                     architecture=ais_parameters["architecture"],
                     add_bias_neuron=True)
    ai.end_act_func = ai.kit_act_func.softmax
    ai.make_all_for_q_learning(("left", "right", "up", "down"),
                               ais_parameters["func_update_q_table"],
                               gamma=ais_parameters["gamma"],
                               epsilon=ais_parameters["epsilon"],
                               q_alpha=ais_parameters["q_alpha"])
    ai.alpha = ais_parameters["alpha"]
    ai.impulse1 = ais_parameters["impulse1"]
    ai.impulse2 = ais_parameters["impulse2"]

    learn_iteration = 0
    while True:
        learn_iteration += 1

        # Выводим максимальный и средний счёт змейки за max_learn_iteration шагов
        if learn_iteration == ais_parameters["max_learn_iteration"]:
            learn_iteration = 0
            _, mean = snake.get_max_mean_score()

            # Лучшая Змейка найдена
            if mean > ais_parameters["threshold_mean_score"]:
                print("ЛУЧШАЯ ЗМЕЙКА НАЙДЕНА!!!", mean)
                ai.update(f"BEST_SNAKE_{round(mean, 1)}")

                found_best_snake.put(True)

        # Записываем данные которые видит Змейка
        data = snake.get_blocks(ais_parameters["visibility_range"])
        # data = snake.get_ranges_to_blocks()

        action = ai.q_predict(data)
        reward = snake.step(action)
        # Обучаем
        ai.q_learning(data, reward, learning_method=ais_parameters["learning_method"],
                      squared_error=ais_parameters["squared_error"],
                      use_Adam=ais_parameters["use_Adam"])


# Создаём сразу много отдельных скриптов
if __name__ == "__main__":
    # Количество одновременно запущенных интерпретаторов (ограничивается количеством ядер)
    amount_ais_to_learning = 3

    # Когда ИИшка достигнет и превысит этот порог, то останавливаем всё обучение
    ais_parameters["threshold_mean_score"] = 1

    found_best_snake = Queue()
    processes = []
    for i in range(amount_ais_to_learning):
        print(f"Process {i} are started")
        process = Process(target=script_learns,
                          args=(snake_parameters,
                                ais_parameters,
                                found_best_snake))

        process.start()
        processes.append(process)

    # Когда лучшая ИИшка будет найдена, останавливаем обучение других
    # (т.е. этот скрипт можно оставить на ночь и на утро будет готовая нейронка)
    if found_best_snake.get() is True:
        for proc in processes:
            proc.terminate()

    print(int(time() - start_time))
    exit()