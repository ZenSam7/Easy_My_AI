from easymyai import AI_ensemble
from Games import Snake
from multiprocessing import Process, Queue
from multiprocessing.managers import BaseManager
from time import time, sleep
import optuna
import pickle

start_time = time()

# Параметры, которые мы вставим во все скрипты
snake_parameters = {
    "wight": 7,
    "height": 5,
    "max_steps": 100,

    "dead_reward": -10,
    "win_reward":  10,

    "amount_food": 1,
}

ais_parameters = {
    "amount_ais": 3,
    "architecture": [9, 100, 100, 100, 4],

    "alpha": 1e-3,
    "impulse1": 0.7,
    "impulse2": 0.9,

    "gamma": .6,
    "epsilon": .0,
    "q_alpha": .1,

    "func_update_q_table": AI_ensemble(1).kit_upd_q_table.future,
    "max_learn_iteration": 50_000,
    "learning_method": 1,
    "visibility_range": 3,
    "squared_error": False,
    "use_Adam": False,
}
# Оптимизируем параметры каждые ... шагов
ais_parameters["num_steps_before_reset"] = 20 * ais_parameters["max_learn_iteration"]


def script_learns(ais_parameters, snake_parameters):
    """апускаем обучение одной нейронки, и возвращаем лечший средний счёт"""

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
    all_means_score = []
    while learn_iteration < ais_parameters["num_steps_before_reset"]:
        learn_iteration += 1

        # Выводим максимальный и средний счёт змейки за max_learn_iteration шагов
        if learn_iteration % ais_parameters["max_learn_iteration"] == 0:
            _, mean = snake.get_max_mean_score()
            all_means_score.append(mean)

            # Лучшая Змейка найдена
            if mean > ais_parameters["threshold_mean_score"]:
                print("ЛУЧШАЯ ЗМЕЙКА НАЙДЕНА!!!", round(mean, 1))
                ai.update(f"BEST_SNAKE_{round(mean, 1)}")

        # Записываем данные которые видит Змейка
        data = snake.get_blocks(ais_parameters["visibility_range"])
        # data = snake.get_ranges_to_blocks()

        action = ai.q_predict(data)
        reward = snake.step(action)
        # Обучаем
        ai.q_learning(data, reward, learning_method=ais_parameters["learning_method"],
                      squared_error=ais_parameters["squared_error"],
                      use_Adam=ais_parameters["use_Adam"])

    # Возвращаем лучший резальтат
    return max(all_means_score)


def select_parameters(trial):
    """Подбираем гиперпараметры при помощи Optuna"""
    ais_local_parameters = ais_parameters.copy()
    snake_local_parameters = snake_parameters.copy()

    """Создаём ИИшку"""
    # Архитектура
    depth = trial.suggest_int("depth", 2, 4)
    widht = trial.suggest_int("width", 50, 200, step=50)
    ais_local_parameters["architecture"] = [9] + [widht] * depth + [4]

    # Коэффициенты
    ais_local_parameters["alpha"] = trial.suggest_float("alpha", 5e-4, 5e-3, log=True)
    ais_local_parameters["gamma"] = trial.suggest_float("gamma", 0, 0.9, step=0.1)
    ais_local_parameters["epsilon"] = trial.suggest_categorical("epsilon", (0, 0.01))

    # Остальное
    ais_local_parameters["squared_error"] = trial.suggest_categorical("squared_error", (True, False))
    ais_local_parameters["use_Adam"] = trial.suggest_categorical("use_Adam", (True, False))

    name_func_update_q_table = trial.suggest_categorical("func_update_q_table",
                                                         ("standart", "future"))
    ais_local_parameters["func_update_q_table"] = getattr(AI_ensemble(1).kit_upd_q_table,
                                                          name_func_update_q_table)

    # Змейка
    snake_local_parameters["dead_reward"] = trial.suggest_int("dead_reward", -20, -5, step=5)
    snake_local_parameters["win_reward"] = trial.suggest_int("win_reward", 5, 20, step=5)

    return script_learns(ais_local_parameters, snake_local_parameters)


def start_selecting_parameters(num_thread: int):
    # Загружаем историю Optuna (если есть)
    study = optuna.create_study(
        study_name=f"Optuna_Saves\\AI_for_Snake_{num_thread}", load_if_exists=True,
    )

    # Каждый запуск оптимизации параметров мы сохраняем историю
    while True:
        study.optimize(select_parameters, n_trials=1)

        # Сохраняем историю Optuna
        with open(f"Optuna_Saves\\AI_for_Snake_{num_thread}.pkl", "wb") as fout:
            pickle.dump(study.sampler, fout)
        print(f"Прошло {int(time() - start_time)} секунд ({int((time() - start_time) // 60)} минут)")


# Создаём сразу много отдельных скриптов
if __name__ == "__main__":
    # Количество одновременно запущенных интерпретаторов (ограничивается количеством ядер)
    amount_threads = 5

    # Когда ИИшка достигнет этот порог средних очков, то сохраняем её
    ais_parameters["threshold_mean_score"] = 16

    processes = []
    for i in range(amount_threads):
        print(f"Process {i} are started")

        process = Process(target=start_selecting_parameters, args=(i,))
        process.start()
        processes.append(process)
