import numpy as np
import json
import os


class AI:
    """Набор функций для работы с самодельным ИИ"""

    def __init__(self):
        self.name = str(np.random.randint(2**31))
        self.weights = []        # Появиться после вызова create_weights
        self.architecture = []   # Появиться после вызова create_weights
        self.moments = 0         # Сохраняем значение предыдущего импульса
        self.velocitys = 0       # Сохраняем значение предыдущей скорости

        # Какую функцию активации используем
        self.kit_act_funcs = self.ActivationFunctions()
        self.what_act_func = self.kit_act_funcs.ReLU_2

        # Альфа коэффициент (коэффициент скорости обучения) (настраивается самостоятельно)
        self.alpha = 1e-3
        # Определяет наличие нейрона смещения (True или False)
        self.have_bias_neuron = True
        # Какую долю нейронов "отключаем" при обучении
        self.number_disabled_weights = 0.0
        # Используем ли Softmax в конце
        self.add_softmax = False

        # Как много градиентов будем усреднять
        self.batch_size = 1
        # Где храним все градиенты
        self.packet_gradients = []

        self.type_error = 1

        self.gamma = 0
        self.epsilon = 0
        self.q_alpha = 0
        self.recce_mode = False

        self.q = []
        self.actions = []
        self.states = []
        self.last_state = []

    def create_weights(
        self, architecture: list, add_bias_neuron=True, min_weight=-1, max_weight=1
    ):
        """Создаёт матрицу со всеми весами между всеми элементами
        (Подавать надо список с количеством нейронов на каждом слое (архитектуру нейронки))
        """

        self.have_bias_neuron = add_bias_neuron
        self.architecture = architecture
        self.weights = []

        # Добавляем все веса между слоями нейронов
        for i in range(len(architecture) - 1):
            self.weights.append(
                self.kit_act_funcs.normalize(
                    np.random.random(
                        size=(architecture[i] + add_bias_neuron, architecture[i + 1])
                    ),
                    min_weight,
                    max_weight,
                )
            )

        self.moments = [0 for _ in range(len(architecture))]
        self.velocitys = [0 for _ in range(len(architecture))]

    def genetic_crossing_with(self, ai):
        """Перемешивает веса между ЭТОЙ нейронкой и нейронкой В АРГУМЕНТЕ \n
        P.s. Не обязательно, чтобы количество связей (размеры матриц весов) были одинаковы
        """

        for layer1, layer2 in zip(self.weights, ai.weights):
            # Для каждого элемента...
            for _ in range(layer1.shape[0] * layer1.shape[1]):
                if np.random.random() < 0.5:  # ... С шансом 50% ...
                    # ... Производим замену на вес из другой матрицы
                    layer1[
                        np.random.randint(layer1.shape[0]),
                        np.random.randint(layer1.shape[1]),
                    ] = layer2[
                        np.random.randint(layer2.shape[0]),
                        np.random.randint(layer2.shape[1]),
                    ]

    def get_mutations(self, mutation=0.05):
        """Создаёт рандомные веса в нейронке"""

        for layer in self.weights:  # Для каждого слоя
            for _ in range(layer.shape[0] * layer.shape[1]):  # Для каждого элемента
                if np.random.random() <= mutation:  # С шансом mutation
                    # Производим замену на случайное число
                    layer[
                        np.random.randint(layer.shape[0]),
                        np.random.randint(layer.shape[1]),
                    ] = (
                        np.random.random() - np.random.random()
                    )

    def start_work(self, input_data: list, return_answers=False):
        """Возвращает результат работы нейронки, из входных данных"""
        # Определяем входные данные как вектор
        result_layer_neurons = np.array(input_data)

        # Сохраняем список всех ответов от нейронов каждого слоя
        list_answers = []

        # Проходимся по каждому (кроме последнего) слою весов
        for layer_weight in self.weights:
            # Если есть нейрон смещения, то в правую часть матриц
            # result_layer_neurons добавляем единицы
            # Чтобы можно было умножить единицы на веса нейрона смещения
            if self.have_bias_neuron:
                result_layer_neurons = np.array(result_layer_neurons.tolist() + [1])

            if return_answers:
                list_answers.append(result_layer_neurons)

            # Процеживаем через функцию активации  ...
            # ... Результат перемножения результата прошлого слоя на слой весов
            # И умножаем рандомные веса на 0 (+ компенсируем нули увеличением остальных весов)
            result_layer_neurons = self.what_act_func(
                result_layer_neurons.dot(layer_weight
                    * (
                        np.random.random(size=layer_weight.shape)
                        >= self.number_disabled_weights
                      )
                    * (1 / (1- self.number_disabled_weights))
                )
            )

        # Добавляем в конец Softmax
        # (надо сделать именно так, чтобы при обратном распространении Softmax не учитывался)
        if self.add_softmax:
            result_layer_neurons = self.kit_act_funcs.Softmax(result_layer_neurons)

        # Если надо, возвращаем список с ответами от каждого слоя
        if return_answers:
            return result_layer_neurons, list_answers

        return result_layer_neurons

    def q_start_work(self, input_data: list):
        """Возвращает action, на основе входных данных"""

        ai_result = self.start_work(input_data)

        # "Разведуем окружающую среду"
        if (np.random.random() < self.epsilon or self.recce_mode):
            # С вероятностью epsilon выбираем случайное действие
            ai_result = np.random.random(ai_result.shape)

        # Находим действие
        return self.actions[np.argmax(ai_result)]

    def learning(
        self,
        input_data: list,
        answer: list,
        get_error=False,
        type_error="regular",
        type_regularization="quadratic",
        regularization_value=10,
        regularization_coefficient=0.1,
        impulse_coefficient=0.9,
    ):
        """Метод обратного распространения ошибки для изменения весов в нейронной сети \n
        Ошибки могут быть: \n
        regular:     линейными \n
        quadratic:   квадратичными \n
        logarithmic: логарифмическими \n
    ------------------ \n

        Регуляризация может быть: \n
        quadratic: Чем больше веса, тем больше увеличиваем ошибку (во квадратичной зависимости)\n
        penalty:   Если веса выходят за границу, то пропорционально увелициваем ошибку \n
    ------------------ \n

        regularization_value: В какои диапазоне (±) держим веса \n
    ------------------ \n

        regularization_coefficient: Как сильно следим за регуляризацией весов \n
    ------------------ \n

        impulse_coefficient: 0 < x < 1, Этот коэффициент влияет на «запоминание» направления градиента\
        движения (направление, в котором веса ИИ изменяются (к минимальной ошибке)) (обычно около 0,9)"""

        # Нормализуем веса (очень грубо)
        if np.any(
                [np.any(np.abs(i) >= 1e6) for i in self.weights]
        ):  # Если запредельные значения весов
            # То пересоздаём веса
            self.create_weights(self.architecture, self.have_bias_neuron)

            # И уменьшаем alpha
            self.alpha /= 10

        # Определяем наш ответ как вектор
        answer = np.array(answer)
        # Определяем наши входные данные как вектор
        input_data = np.array(input_data)

        # То, что выдала нам нейросеть | Список с ответами от каждого слоя нейронов
        ai_answer, answers = self.start_work(input_data, True)

        # На сколько должны суммарно изменить веса
        delta_weight = ai_answer - answer
        if type_error == "quadratic":
            delta_weight = np.power(delta_weight, 2) * (
                    -1 * (delta_weight < 0) + 1 * (delta_weight >= 0)
            )  # Тут сохраняем знак
        elif type_error == "logarithmic":
            delta_weight = np.power(np.log(ai_answer - answer + 1), 2) * (
                    -1 * (delta_weight < 0) + 1 * (delta_weight >= 0)
            )  # Тут сохраняем знак

        # Регуляризация (держим веса близкими к 0, чтобы не улетали в космос)
        if type_regularization == "quadratic":
            delta_weight += (
                    (-1 * (delta_weight < 0) + 1 * (delta_weight >= 0))
                    * regularization_coefficient
                    * np.sqrt(
                np.sum([np.sum(np.power(i / regularization_value, 2))
                        for i in self.weights])
            ))
        elif type_regularization == "penalty":
            delta_weight += (
                    (-1 * (delta_weight < 0) + 1 * (delta_weight >= 0))
                    * regularization_coefficient
                    * np.sum([
                np.sum(
                    np.abs(i * (np.abs(i) >= regularization_value))
                    - regularization_value
                )
                for i in self.weights]
            ))

        delta_weight = np.matrix(delta_weight)  # Превращаем вектор в матрицу

        # Совершаем всю магию здесь
        list_gradients = []
        for weight, layer_answer, moment, velocity in zip(
                self.weights[::-1],
                answers[::-1],
                self.moments[::-1],
                self.velocitys[::-1],
        ):
            # Превращаем вектор в матрицу
            delta_weight = np.matrix(delta_weight)
            layer_answer = np.matrix(layer_answer)

            # Изменяем веса с оптимизацией Adam
            gradient = (delta_weight.T.dot(layer_answer)).T
            moment = impulse_coefficient * moment + (1 - impulse_coefficient) * gradient
            velocity = impulse_coefficient * velocity + \
                       (1 - impulse_coefficient) * np.power(gradient, 2)

            MOMENT = moment / (1 - impulse_coefficient)
            VELOCITY = velocity / (1 - impulse_coefficient)

            # "Переносим" на другой слой и умножаем на производную
            delta_weight = np.multiply(
                delta_weight.dot(weight.T), self.what_act_func(layer_answer, True)
            )

            # Записываем градиент
            list_gradients.append(MOMENT / (np.sqrt(VELOCITY) + 1))
            # weight -= self.alpha * (MOMENT / (np.sqrt(VELOCITY) + 1))

            # К нейрону смещения не идут связи, поэтому обрезаем этот нейрон смещения
            if self.have_bias_neuron == True:
                delta_weight = np.matrix(delta_weight.T.tolist()[0:-1]).T


        self.packet_gradients.append(list_gradients[::-1])

        # Изменяем веса
        if len(self.packet_gradients) == self.batch_size:
            # Усредняем градиенты (если надо)
            if self.batch_size != 1:
                list_gradients = []
                for num_layer_weight in range(len(self.weights)):
                    # Усредняем градиенты поочереди
                    # (сначала все градиенты для первого слоя весов,
                    #   потом для второго слоя весов...)
                    mean_grads = np.mean(
                        [grads[num_layer_weight] for grads in self.packet_gradients],
                        axis=0
                    )

                    # Добавляем усреднённый градиент
                    list_gradients.append(mean_grads)

            else:
                list_gradients = self.packet_gradients[0]

            # Наконец-то изменяем веса
            for weight, gradient in zip(self.weights, list_gradients):
                # Увеличиваем alpha, т.к. мы реже изменяем веса, а значит
                # обучение проиходит медленнее
                weight -= (self.alpha * self.batch_size) * gradient

            self.packet_gradients.clear()

    def make_all_for_q_learning(
        self, actions: list, gamma=0.5, epsilon=0.0, q_alpha=0.1
    ):
        """Создаём всё необходимое для Q-обучения \n
        Q-таблицу (таблица вознаграждений за действие) \n
        Коэффициент важности будущего вознаграждения gamma \n
        Коэффициент почти случайных действий epsilon \n
        Коэффициент скорости изменения Q-таблицы q_alpha"""

        self.actions = actions
        self.gamma = gamma  # Коэффициент на сколько важно будущее вознаграждение
        self.epsilon = epsilon  # Коэффициент "разведки окружающей среды"
        self.q_alpha = q_alpha

        # Таблица состояний (заполняем нулевым состоянием)
        # Размеры: States ⨉ Actions
        # # States — количество уникальных состояний (С каждым новым состояние пополняется)
        self.q = [[0 for _ in range(len(actions))]]

        # Заполняем "первое" (несуществующее (т.к. мы в прошлом на 1 шаг))
        # состояние количеством входов
        self.states = [
            [-0.0 for _ in range(self.weights[0].shape[0] - self.have_bias_neuron)]
        ]
        # Прошлое состояние нужно для откатывания на 1 состояние назад
        self.last_state = self.states[0]
        self.last_reward = 0

    def q_learning(
        self,
        state,
        reward_for_state,
        num_update_function=1,
        learning_method=2.2,
        type_error="regular",
        recce_mode=False,
        type_regularization="quadratic",
        regularization_value=2,
        regularization_coefficient=0.1,
        impulse_coefficient=0.9,
    ):
        """ Глубокое Q-обучение (ИИ используется как предсказатель правильных действий)

-------------------------- \n

        num_function - это номер функции обновления Q-таблицы: \n
        \n 1: Q(s,a) = Q(s,a) + α[r + γ(max Q(s’,a')) - Q(s,a)] \n
        \n 2: Q(s,a) = Q(s,a) + α[r + γ Q(s’,a') - Q(s,a)] \n
        \n 3: Q(s,a) = Q(s,a) + α[r + γ(Expected Q(s’,a')) - Q(s,a)] \n
        \n 4: Q(s,a) = R + γ Q’(s’,a’) \n
        \n 5: Q(s,a) = R + γ Q’(s’, max a) \n

-------------------------- \n

        Ошибки могут быть: \n
        regular: |ai_answer - answer| \n
        quadratic: (ai_answer - answer)^2 \n
        logarithmic: ln^2( (ai_answer - answer) +1 ) \n

-------------------------- \n

        recce_mode - при значении True включаем "режим разведки", т.е. при таком режиме ИИ не обучается,\
        а только пополняется Q-таблица

-------------------------- \n

        Методы обучения (значение learning_method определяет) : \n
        1 : В качестве "правильного" ответа выбирается то, которое максимально вознаграждается, и на место действия \
        (которое приводит к лучшему ответу) ставиться максимальное значение функции активации \
        (self.kit_act_funcs.max), а на остальные места минимум функции активации (self.kit_act_funcs.min) \n
        P.s. Это неочень хорошо, т.к. игнорируются другие варианты, которые приносят либо столько же, либо немного меньше  \
        вознаграждения (а выбирается только один "правильный"). НО ОН ХОРОШО ПОДХОДИТ,\
        КОГДА У ВАС В ЗАДАЧЕ ИМЕЕТСЯ ИСКЛЮЧИТЕЛЬНО 1 ПРАВИЛЬНЫЙ ОТВЕТ, А "БОЛЕЕ" И "МЕНЕЕ" ПРАВИЛЬНЫХ БЫТЬ НЕ МОЖЕТ \n
        \n

        2 : Делаем ответы которые больше вознаграждаются, более "правильным" \n
        Дробная часть числа означает, в какую степень будем возводить "стремление у лучшим результатам"\
        (что это такое читай в P.s.) \
        (чем степень больше, тем этот режим будет больше похож на режим 1. НАПРИМЕР: 2.2 означает, что мы \
        используем метод обучения 2 и возводим в степень 2 "стремление у лучшим результатам",\
        а 2.345 означает, что степень будет равна 3.45 ) \n
        P.s. Работает так: Сначала переводим значения вознаграждений в промежуток от 0 до 1 (т.е. где вместо максимума вознаграждения\
        - 1, в место минимума - 0, а остальные вознаграждения между ними (без потерь "расстояний" между числами)) \
        потом прибавляем 0.5 и возводим в степень "стремление у лучшим результатам" (уже искажаем "расстояние" между числами) \
        (чтобы ИИ больше стремился именно к лучшим результатам и совсем немного учитывал остальные \
        (чем степень больше, тем меньше учитываются остальные результаты))  \n
        """

        STATE = self.states.index(self.last_state)
        self.recce_mode = recce_mode

        # Q-обучение

        # Формируем "правильный" ответ
        answer = [0 for _ in range(len(self.actions))]
        if learning_method == 1:
            answer = [self.kit_act_funcs.min for _ in range(len(self.actions))]

            # На месте максимального значения из Q-таблицы ставим максимально
            # возможное значение как "правильный" ответ
            answer[self.q[STATE].index(max(self.q[STATE]))] = self.kit_act_funcs.max

        elif 2 < learning_method < 3:
            # Искажаем "расстояние" между числами
            answer = self.kit_act_funcs.normalize(np.array(self.q[STATE]), 0, 2)
            answer = np.power(answer, (learning_method - 2) * 10)

            # Переводим в промежуток от min до max
            answer = self.kit_act_funcs.normalize(
                np.array(self.q[STATE]), self.kit_act_funcs.min, self.kit_act_funcs.max
            )

            answer = answer.tolist()

        # Если режим разведки выключен
        if self.recce_mode == False:
            # Обновляем Q-таблицу
            self._update_q_table(state, reward_for_state, num_update_function)

            # Изменяем веса (не забываем, что мы находимся в состоянии на 1 шаг
            # назад)
            self.learning(
                self.last_state,
                answer,
                type_error=type_error,
                type_regularization=type_regularization,
                regularization_value=regularization_value,
                regularization_coefficient=regularization_coefficient,
                impulse_coefficient=impulse_coefficient,
            )

        else:
            # Иначе просто обновляем таблицу с изменёнными параметрами
            Gamma, Epsilon, Q_alpha = self.gamma, self.epsilon, self.q_alpha
            self.gamma, self.epsilon, self.q_alpha = 0.01, 2, 0.01

            self._update_q_table(state, reward_for_state, num_update_function)

            self.gamma, self.epsilon, self.q_alpha = Gamma, Epsilon, Q_alpha

    def _update_q_table(self, state, reward_for_state, num_update_function):
        """Формулы для обновления Q-таблицы \n

        --------------------------

        \n 1: Q(s,a) = Q(s,a) + α[r + γ(max Q(s’,a')) - Q(s,a)] \n
        \n 2: Q(s,a) = Q(s,a) + α[r + γ Q(s’,a') - Q(s,a)] \n
        \n 3: Q(s,a) = Q(s,a) + α[r + γ(Expected Q(s’,a')) - Q(s,a)] \n
        \n 4: Q(s,a) = R + γ Q’(s’,a’) \n
        \n 5: Q(s,a) = R + γ Q’(s’, max a) \n
        """

        # Если не находим состояние в прошлых состояниях (Q-таблице), то
        # добавляем новое
        if not state in self.states:
            self.states.append(state)
            self.q.append([0 for _ in range(len(self.actions))])

        # Откатываем наше состояние на 1 шаг назад
        # (Текущее (по реальному времени) == Будущее, а Прошлое == Настоящее)
        REWARD = self.last_reward
        # "Текущее" на самом деле прошлое
        STATE = self.states.index(self.last_state)
        # "Будущее" на самом деле настоящее
        FUTURE_STATE = self.states.index(state)

        # С учётом выше написанного, наше "текущее" действие == ответ нейронки
        # на прошлое состояние
        if self.epsilon < 1:
            ACT = self.actions.index(self.q_start_work(self.last_state))
            FUTURE_ACT = self.actions.index(self.q_start_work(state))
        else:
            ACT = np.random.randint(len(self.actions))  # Случайное действие
            FUTURE_ACT = np.random.randint(len(self.actions))  # Случайное действие

        # А "прошлое" уже является настоящим (т.к. все переменные уже
        # объявлены)
        self.last_state = state
        self.last_reward = reward_for_state

        if num_update_function == 1:
            self.q[STATE][ACT] = self.q[STATE][ACT] + self.q_alpha * (
                REWARD + self.gamma * max(self.q[FUTURE_STATE]) - self.q[STATE][ACT]
            )

        elif num_update_function == 2:
            self.q[STATE][ACT] = self.q[STATE][ACT] + self.q_alpha * (
                REWARD
                + self.gamma * self.q[FUTURE_STATE][FUTURE_ACT]
                - self.q[STATE][ACT]
            )

        elif num_update_function == 3:
            self.q[STATE][ACT] = self.q[STATE][ACT] + self.q_alpha * (
                REWARD + self.gamma * sum(self.q[FUTURE_STATE]) - self.q[STATE][ACT]
            )

        elif num_update_function == 4:
            self.q[STATE][ACT] = REWARD + self.gamma * self.q[FUTURE_STATE][FUTURE_ACT]

        elif num_update_function == 5:
            self.q[STATE][ACT] = REWARD + self.gamma * max(self.q[FUTURE_STATE])

    def save(self, ai_name=""):
        """Сохраняет всю необходимую информацию о текущей ИИ"""

        name_ai = self.name if ai_name == "" else ai_name

        # Записываем данны об ИИшке
        def get_name_act_func(func):
            func_str = str(func)

            if "ReLU_2" in func_str:
                return "ReLU_2"
            elif "ReLU" in func_str:
                return "ReLU"
            elif "Softmax" in func_str:
                return "Softmax"
            elif "Tanh" in func_str:
                return "Tanh"
            elif "Sigmoid" in func_str:
                return "Sigmoid"

        ai_data = {}

        ai_data["weights"] = [i.tolist() for i in self.weights]
        ai_data["architecture"] = self.architecture
        ai_data["have_bias_neuron"] = self.have_bias_neuron

        ai_data["number_disabled_weights"] = self.number_disabled_weights
        ai_data["alpha"] = self.alpha
        ai_data["batch_size"] = self.batch_size

        ai_data["what_act_func"] = get_name_act_func(self.what_act_func)
        ai_data["value_range"] = [self.kit_act_funcs.min, self.kit_act_funcs.max]

        ai_data["q_table"] = self.q

        ai_data["states"] = self.states
        ai_data["last_state"] = self.last_state
        ai_data["actions"] = self.actions

        ai_data["last_reward"] = 0
        ai_data["gamma"] = self.gamma
        ai_data["epsilon"] = self.epsilon
        ai_data["q_alpha"] = self.q_alpha

        with open(f"Saves AIs/{name_ai}.json", "w+") as save_file:
            json.dump(ai_data, save_file)

    def load(self, ai_name=""):
        """Загружает все данные сохранённой ИИ"""

        name_ai = self.name if ai_name == "" else ai_name

        # Записываем данны об ИИшке
        def get_act_func_with_name(name):
            if name == "ReLU_2":
                return self.kit_act_funcs.ReLU_2
            elif name == "ReLU":
                return self.kit_act_funcs.ReLU
            elif name == "Softmax":
                return self.kit_act_funcs.Softmax
            elif name == "Tanh":
                return self.kit_act_funcs.Tanh
            elif name == "Sigmoid":
                return self.kit_act_funcs.Sigmoid

        with open(f"Saves AIs/{name_ai}.json", "r") as save_file:
            ai_data = json.load(save_file)

        self.weights = [np.array(i) for i in ai_data["weights"]]
        self.architecture = ai_data["architecture"]
        self.have_bias_neuron = ai_data["have_bias_neuron"]

        self.number_disabled_weights = ai_data["number_disabled_weights"]
        self.alpha = ai_data["alpha"]
        self.batch_size = ai_data["batch_size"]

        self.what_act_func = get_act_func_with_name(ai_data["what_act_func"])
        self.kit_act_funcs.min, self.kit_act_funcs.max = ai_data["value_range"][0], ai_data["value_range"][1]

        self.q = ai_data["q_table"]

        self.states = ai_data["states"]
        self.last_state = ai_data["last_state"]
        self.actions = ai_data["actions"]

        self.last_reward = ai_data["last_reward"]
        self.gamma = ai_data["gamma"]
        self.epsilon = ai_data["epsilon"]
        self.q_alpha = ai_data["q_alpha"]

    def delete(self, ai_name=""):
        """Удаляет сохранение"""

        name_ai = self.name if ai_name == "" else ai_name

        try:
            os.remove(f"Saves AIs/{name_ai}.json")
        except:
            pass

    def print_how_many_parameters(self):
        parameters = []
        for layer in self.weights:
            parameters.append(layer.shape[0] * layer.shape[1])

        print(f"Parameters: {sum(parameters)} \t\t {self.architecture} ", end="")

        if self.have_bias_neuron:
            print("+ bias", end="")

        print()

    class ActivationFunctions:
        """Набор функций активации и их производных"""

        def __init__(self):
            self.min = 0
            self.max = 1

        def normalize(self, x, min=0, max=1):
            # Нормализуем от 0 до 1
            result = x - np.min(x)
            if np.max(x) != 0:
                result = result / np.max(result)

            # Потом от min до max
            result = result * (max - min) + min

            return result

        def ReLU(self, x, return_derivative=False):
            """Не действует ограничение value_range"""

            if return_derivative:
                return (x > 0) * 1

            else:
                return (x > 0) * x

        def ReLU_2(self, x, return_derivative=False):
            if return_derivative:
                return (
                    (x < 0) * 0.01 +
                    np.multiply(0 <= x, x <= 1) * 1 +
                    (x > 1) * 0.01
                    )

            else:
                return (
                    (x < 0) * 0.01 * x +
                    np.multiply(0 <= x, x <= 1) * x +
                    ((x > 1) * 0.01 * x + 0.99)
                    )

        def Softmax(self, x, return_derivative=False):
            return np.exp(x) / np.sum(np.exp(x))

        def Tanh(self, x, return_derivative=False):
            if return_derivative:
                return 1 / (np.power(np.cosh(x), 2))

            else:
                return np.tanh(x)

        def Sigmoid(self, x, return_derivative=False):
            if return_derivative:
                return np.exp(-1 * x) / (np.power(1 + np.exp(-1 * x), 2))

            else:
                return 1 / (1 + np.exp(-1 * x))
