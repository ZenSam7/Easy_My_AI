from Code_My_AI import AI
from Games import Code_Snake
# from time import time
# from profilehooks import profile
import optuna
import pickle

# start = time()
ai_counter = 0

def end():
    global reward
    reward = -10
def win():
    global reward
    reward = 100

# Создаём Змейку
snake = Code_Snake.Snake(700, 600, 3, 0, max_num_steps=100, display_game=False,
                         game_over_function=end, eat_apple_function=win)


def go_test_ai(trial):
    global ai_counter
    ai_counter += 1

    """Создаём ИИшку"""
    # Архитектура
    depth = trial.suggest_int("depth", 1, 5)
    widht = trial.suggest_int("width", 50, 200)

    architecture = [9] + [widht] * depth + [4]

    ai = AI(architecture=architecture, add_bias_neuron=True, name="Snake")

    ai.what_act_func = ai.kit_act_func.tanh
    ai.end_act_func = ai.kit_act_func.softmax

    # Коэффициенты
    alpha = trial.suggest_float("alpha", 1e-6, 1e-5)
    batch_size = trial.suggest_int("batch_size", 1, 101, step=5)
    squared_error = trial.suggest_categorical("squared_error", (True, False))

    ai.alpha = alpha
    ai.batch_size = batch_size

    # Для Q-обучения
    gamma = trial.suggest_float("gamma", 0, 1)
    epsilon = trial.suggest_float("epsilon", 0, 0.1)
    q_alpha = trial.suggest_float("q_alpha", 0.01, 1)
    learning_method = trial.suggest_categorical("learning_method", (1, 2.99))

    # Создаём функцию обновления Q-таблицы через костыли
    upd_func_name = trial.suggest_categorical("upd_func",
                                              ("standart", "future", "future_sum", "simple", "simple_max"))
    upd_func = getattr(ai.kit_upd_q_table, upd_func_name)

    ai.make_all_for_q_learning(("left", "right", "up", "down"), upd_func,
                               gamma, epsilon, q_alpha)

    # ai.load()
    # ai.print_how_many_parameters()


    """Тестируем ИИшку"""
    for step in range(500_000): # Количество шагов
        # Далее производим 1 итерацию обучения:
        reward = 0

        input_data = snake.get_blocks(3)

        action = ai.q_start_work(input_data)
        snake.step(action)

        # Обучаем
        ai.q_learning(input_data, reward, snake.get_future_state(action),
                      recce_mode=False, learning_method=learning_method, squared_error=squared_error)

        # На 400_000 шаге сбрасываем среднее число очков
        # (т.к. нам надо проверить как ИИшка справляется в уже более-менее
        # обученном виде, а не считая с самого начала)
        if step == 400_000:
            snake.scores.clear()

    # Мы хотим максимизировать среднее число очков
    _, mean = snake.get_max_mean_score()

    return mean


# Создаём новый лист для истории Optuna
snake_ai = optuna.create_study(direction="maximize", study_name="Snake_AI")

# Загружаем историю Optuna
loaded_sampler = pickle.load(open("Snake_AI.pkl", "rb"))
snake_ai = optuna.create_study(
    study_name="Snake_AI", load_if_exists=True, sampler=loaded_sampler
)

for _ in range(100):
    snake_ai.optimize(go_test_ai, n_trials=1)

    # Сохраняем историю Optuna
    with open("Snake_AI.pkl", "wb") as fout:
        pickle.dump(snake_ai.sampler, fout)
