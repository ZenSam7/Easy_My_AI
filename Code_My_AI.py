import numpy as np
import json


class AI:
    """Набор функций для работы с самодельным ИИ"""
    def __init__(self):
        self.name = str(np.random.randint(2**31))
        self.weights = []          # Появиться после вызова create_weights

        self.kit_act_func = self.ActivationFunctions()
        # Какую функцию активации используем
        self.what_act_func = self.kit_act_func.Tanh
        # Какую функцию активации используем для выходных значений
        self.end_act_func  = self.kit_act_func.Tanh

        self.alpha = 1e-2     # Альфа каэффицент (каэффицент скорости обучения)
        self.have_bias_neuron = True      # Определяет наличие нейрона смещения
        self.number_disabled_weights = 0.0      # Какую долю весов "отключаем" при обучении

        # Как много ошибок будем усреднять, чтобы на основе этой усреднённой ошибки изменять веса
        self.batch_size = 1
        self.packet_gradients = []  # Где мы будем эти ошибки складывать
        self.packet_errors = []

        self.q_table = []         # Q-table
        self.states = []    # Все состояния
        self.actions = []     # Все действия
        self.recce_mode = False
        self.gamma = 0.1
        self.epsilon = 0
        self.q_alpha = 0.1

    def create_weights(self, architecture: list, add_bias_neuron=True,
                       min_weight=-1, max_weight=1):
        """Создаёт матрицу со всеми весами между всеми элементами
        (Подавать надо список с количеством нейронов на каждом слое (архитектуру нейронки))"""

        self.have_bias_neuron = add_bias_neuron
        self.architecture = architecture

        # Добавляем все веса между слоями нейронов
        for i in range(len(architecture) - 1):
            self.weights.append(
                self.kit_act_func.normalize(
                    np.random.random(
                        size=(architecture[i] + add_bias_neuron, architecture[i + 1])
                    ),
                    min_weight,
                    max_weight,
                )
            )

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
        for layer_weight in self.weights[:-1]:
            # Если есть нейрон смещения, то в правую часть матриц
            # result_layer_neurons добавляем еденицы
            # Чтобы можно было умножить еденицы на веса нейрона смещения
            if self.have_bias_neuron:
                result_layer_neurons = np.array(result_layer_neurons.tolist() + [1])

            if return_answers:
                list_answers.append(result_layer_neurons)

            # Процежеваем через функцию активации  ...
            # ... Результат перемножения результата прошлого слоя на слой весов
            result_layer_neurons = self.what_act_func(
                                        result_layer_neurons.dot(layer_weight) )


        # Добавляем ответ (единицу) для нейрона смещения, для последнего перемножения
        if self.have_bias_neuron:
            result_layer_neurons = np.array(result_layer_neurons.tolist() + [1])
        if return_answers:
            list_answers.append(result_layer_neurons)


        # Пропускаем выходные данные через последнюю функцию активации (Если есть)
        if self.end_act_func == None:
            result_layer_neurons = result_layer_neurons.dot(self.weights[-1])
        else:
            result_layer_neurons = self.end_act_func(
                        result_layer_neurons.dot(self.weights[-1]))


        # Если надо, возвращаем спосок с ответами от каждого слоя
        if return_answers:
            return result_layer_neurons, list_answers
        else:
            return result_layer_neurons

    def learning(self, input_data: list, answer: list,
                 get_error=False,
                 squared_error=False,
                 use_adam=True):
        """Метод обратного распространения ошибки для изменения весов в нейронной сети \n"""

        # Нормализуем веса (очень грубо)
        if np.any(np.abs(self.weights[0]) >= 1e6):    # Если запредельные значения весов
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
        if squared_error:
            delta_weight = np.power(ai_answer - answer, 2) * \
                           (-1* ((ai_answer - answer) <0) + 1*((ai_answer - answer) >=0) )
                           # Оставляем знак ↑


        # Совершаем всю магию здесь
        list_gradients = []

        for weight, layer_answer in zip(self.weights[::-1], answers[::-1]):
            # Превращаем векторы в матрицу
            layer_answer = np.matrix(layer_answer)
            delta_weight = np.matrix(delta_weight)

            gradients = layer_answer.T.dot(delta_weight)

            # Матрица, предотвращающая переобучение
            # Умножаем изменение веса рандомных нейронов на 0
            if self.number_disabled_weights > 0:
                dropout_mask = np.random.random(size=(layer_answer.shape[1], delta_weight.shape[1])) \
                               >= self.number_disabled_weights

                gradients = np.multiply(dropout_mask, # Отключаем изменение некоторых связей
                                        layer_answer.T.dot(delta_weight))

            # Записываем градиент
            list_gradients.append(gradients)

            # К нейрону смещения не идут связи, поэтому обрезаем этот нейрон смещения
            if self.have_bias_neuron:
                weight = weight[0:-1]
                layer_answer = np.matrix(layer_answer.tolist()[0][0:-1])

            # "Переносим" градиент на другой слой (+ умножаем на производную)
            delta_weight = delta_weight.dot(weight.T)
            delta_weight.dot(self.what_act_func(layer_answer, True).T)


        self.packet_gradients.append(list_gradients[::-1])

        # Изменяем веса
        if len(self.packet_gradients) == self.batch_size:
            # Складываем градиенты (если надо)
            if self.batch_size != 1:
                list_gradients = []
                for num_layer_weight in range(len(self.weights)):
                    # Складываем градиенты поочереди
                    # (сначала все градиенты для первого слоя весов,
                    #   потом для второго слоя весов...)
                    mean_grads = np.sum(
                        [grads[num_layer_weight] for grads in self.packet_gradients],
                        axis=0
                    )

                    # Добавляем суммарный градиент
                    list_gradients.append(mean_grads)

            else:
                list_gradients = self.packet_gradients[0]

            # Наконец-то изменяем веса
            for weight, gradient in zip(self.weights, list_gradients):
                weight -= self.alpha * gradient

            self.packet_gradients.clear()

    def q_start_work(self, input_data: list):
        """Возвращает action, на основе входных данных"""

        ai_result = self.start_work(input_data).tolist()

        # "Разведуем окружающую среду" (берём случайное действие)
        if (self.recce_mode) or (self.epsilon != 0) and (np.random.random() < self.epsilon):
            return self.actions[np.random.randint(len(self.actions))]

        # Находим действие
        return self.actions[np.argmax(ai_result)]

    def make_all_for_q_learning(self, actions: list, gamma=0.1, epsilon=0.1, q_alpha=0.1):
        """Создаём всё необходимое для Q-обучения \n
        Q-таблицу (таблица вознаграждений за действие), каэфицент вознаграждения gamma, \
        каэфицент почти случайных действий epsilon, и каэфицент скорости изменения Q-таблицы q_alpha """

        self.actions = actions
        self.gamma = gamma      # Каэфицент "доверия опыту"
        self.epsilon = epsilon  # Каэфицент "разведки окружающей среды"
        self.q_alpha = q_alpha

        self.q_table = []    # Таблица состояний

    def q_learning(self, state, reward_for_state, future_state,
                   num_update_function=1,
                   learning_method=2.1,
                   squared_error=False,
                   recce_mode=False,
                   ):
        """
        ИИ используется как предсказатель правильных действий\n

        --------------------------

        num_function - это номер функции обновления Q-таблицы: \n
        \n 1: Q(s,a) = Q(s,a) + α[r + γ(max Q(s’,a')) - Q(s,a)] \n
        \n 2: Q(s,a) = (1 - α) Q(s,a) + α[r + γ(max Q(s’,a))] \n
        \n 3: Q(s,a) = Q(s,a) + α[r + γ Q(s’,a') - Q(s,a)] \n
        \n 4: Q(s,a) = Q(s,a) + α[r + γ(Expected Q(s’,a')) - Q(s,a)] \n
        \n 5: Q(s,a) = R + γ Q’(s’,a’) \n
        \n 6: Q(s,a) = R + γ Q’(s’, max a) \n

        \n

        -------------------------- \n

        Методы обучения (значение learning_method определяет) : \n
        1 : В качестве "правильного" ответа выбирается то, которое максимально вознаграждается, и на место действия \
        (которое приводит к лучшему ответу) ставиться максимальное значение функции активации \
        (self.activation_function.max), а на остальные места минимум функции активации (self.activation_function.min) \n
        P.s. Это неочень хорошо, т.к. игнорируются другие варианты, которые приносят либо столько же,
        либо немного меньше  \
        вознаграждения (а вибирается только один "правильный"). НО ОН ХОРОШО ПОДХОДИТ,
        КОГДА У ВАС В ЗАДАЧЕ ИМЕЕТСЯ ИСКЛЮЧИТЕЛЬНО 1 \
        ПРАВИЛЬНЫЙ ОТВЕТ, А "БОЛЕЕ" И "МЕНЕЕ" ПРАВИЛЬНЫХ БЫТЬ НЕ МОЖЕТ \n
        \n

        2 : Делаем ответы которые больше вознаграждаются, более "правильным" \n
        Дробная часть числа означает, в какую степень будем возводить "стремление у лучшим результатам"
        (что это такое читай в P.s.) \
        (чем степень больше, тем степень будет больше. НАПРИМЕР: 2.2 означает, что мы используем
        метод обучения 2 и возводим в степень 2 \
        "стремление у лучшим результатам", а 2.345 означает, что степень будет равна 3.45 ) \n
        P.s. Работает так: Сначала переводим значения вознаграждений в промежуток от 0 до 1
        (т.е. где вместо максимума вознаграждения\
        - 1, в место минимума - 0, а остальные вознаграждения между ними (без потерь "расстояний" между числами)) \
        потом прибавляем 0.5 и возводим в степень "стремление у лучшим результатам" (уже искажаем "расстояние" между числами) \
        (чтобы ИИ больше стремился именно к лучшим результатам и совсем немного учитывал остальные \
        (чем степень больше, тем меньше учитываются остальные результаты))
        """

        # Если не находим состояние в прошлых состояниях (Q-таблице), то добовляем новое
        if not state in self.states:
            self.states.append(state)
            self.q_table.append([0 for _ in range(len(self.actions))])
        if not future_state in self.states:
            self.states.append(future_state)
            self.q_table.append([0 for _ in range(len(self.actions))])

        STATE = self.states.index(state)
        self.recce_mode = recce_mode


        # Q-обучение

        # Формируем "правильный" ответ
        if learning_method == 1:
            answer = [0 for _ in range(len(self.actions))]

            # На месте максимального значения из Q-таблицы ставим
            # максимально возможное значение как "правильный" ответ
            answer[self.q_table[STATE].index(max(self.q_table[STATE]))] = 1

        elif 2 < learning_method < 3:
            # Нам нужны значения от минимума функции активации до максимума функции активации
            answer = self.kit_act_func.normalize(np.array(self.q_table[STATE]))

            # Искажаем "расстояние" между числами
            answer = answer + 0.5
            answer = np.power(answer, (learning_method -2) *10 )

            answer = self.kit_act_func.normalize(answer).tolist()

        if recce_mode:
            Epsilon, self.epsilon = self.epsilon, 1
            self._update_q_table(state, reward_for_state, num_update_function)
            self.epsilon = Epsilon

        else:
            # Обновляем Q-таблицу
            self._update_q_table(state, reward_for_state, future_state, num_update_function)
            # Изменяем веса
            self.learning(state, answer, squared_error=squared_error)

    def _update_q_table(self, state, reward_for_state, future_state, num_function):
        """Формулы для обновления Q-таблицы \n

        --------------------------

        \n 1: Q(s,a) = Q(s,a) + α[r + γ(max Q(s’,a')) - Q(s,a)] \n
        \n 2: Q(s,a) = (1 - α) Q(s,a) + α[r + γ(max Q(s’,a))] \n
        \n 3: Q(s,a) = Q(s,a) + α[r + γ Q(s’,a') - Q(s,a)] \n
        \n 4: Q(s,a) = Q(s,a) + α[r + γ(Expected Q(s’,a')) - Q(s,a)] \n
        \n 5: Q(s,a) = R + γ Q’(s’,a’) \n
        \n 6: Q(s,a) = R + γ Q’(s’, max a) \n
        """
        action = self.q_start_work(state)
        state = [i for i in state]

        STATE = self.states.index(state)
        ACT = self.actions.index(action)

        future_state = [i for i in future_state]


        if num_function == 1:
            self.q_table[STATE][ACT] = self.q_table[STATE][ACT] + \
                                       self.q_alpha * (reward_for_state + \
                                           self.gamma * max(self.q_table[self.states.index(future_state)]) - \
                                           self.q_table[STATE][ACT])

        elif num_function == 2:
            self.q_table[STATE][ACT] = (1 - self.q_alpha) * self.q_table[STATE][ACT] + \
                                       self.q_alpha * (reward_for_state + \
                                           self.gamma * max(self.q_table[self.states.index(future_state)]))

        elif num_function == 3:
            self.q_table[STATE][ACT] = self.q_table[STATE][ACT] + self.q_alpha * (reward_for_state + \
                                          self.gamma *
                                          self.q_table[self.states.index(future_state)][self.actions.index(
                                          self.q_start_work(future_state))] - \
                                          self.q_table[STATE][ACT])

        elif num_function == 4:
            self.q_table[STATE][ACT] = self.q_table[STATE][ACT] + \
                                       self.q_alpha * (reward_for_state + \
                                       self.gamma * sum(self.q_table[self.states.index(future_state)]) - \
                                       self.q_table[STATE][ACT])

        elif num_function == 5:
            self.q_table[STATE][ACT] = reward_for_state + self.gamma * \
                                       self.q_table[self.states.index(future_state)][self.actions.index(
                                        self.q_start_work(future_state))]

        elif num_function == 6:
            self.q_table[STATE][ACT] = reward_for_state + self.gamma *\
                                       max(self.q_table[self.states.index(future_state)])

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

        ai_data["q_table"] = self.q_table
        ai_data["states"] = self.states
        # ai_data["last_state"] = self.last_state
        ai_data["actions"] = self.actions

        ai_data["architecture"] = self.architecture
        ai_data["have_bias_neuron"] = self.have_bias_neuron

        ai_data["number_disabled_neurons"] = self.number_disabled_weights
        ai_data["alpha"] = self.alpha
        ai_data["batch_size"] = self.batch_size

        ai_data["what_activation_function"] = get_name_act_func(self.what_act_func)
        ai_data["end_activation_function"] = get_name_act_func(self.end_act_func)

        # ai_data["last_reward"] = 0
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
                return self.kit_act_func.ReLU_2
            elif name == "ReLU":
                return self.kit_act_func.ReLU
            elif name == "Softmax":
                return self.kit_act_func.Softmax
            elif name == "Tanh":
                return self.kit_act_func.Tanh
            elif name == "Sigmoid":
                return self.kit_act_func.Sigmoid

        try:
            with open(f"Saves AIs/{name_ai}.json", "r") as save_file:
                ai_data = json.load(save_file)

            self.weights = [np.array(i) for i in ai_data["weights"]]
            self.architecture = ai_data["architecture"]
            self.have_bias_neuron = ai_data["have_bias_neuron"]

            self.number_disabled_weights = ai_data["number_disabled_neurons"]
            self.alpha = ai_data["alpha"]
            self.batch_size = ai_data["batch_size"]

            self.what_act_func = get_act_func_with_name(ai_data["what_activation_function"])
            self.end_act_func = get_act_func_with_name(ai_data["end_activation_function"])

            self.q_table = ai_data["q_table"]

            self.states = ai_data["states"]
            # self.last_state = ai_data["last_state"]
            self.actions = ai_data["actions"]

            # self.last_reward = ai_data["last_reward"]
            self.gamma = ai_data["gamma"]
            self.epsilon = ai_data["epsilon"]
            self.q_alpha = ai_data["q_alpha"]

        except FileNotFoundError:
            raise f"Сохранение {name_ai} не найдено"

    def delete(self, ai_name=""):
        """Удаляет сохранение"""

        name_ai = self.name if ai_name == "" else ai_name

        try:
            os.remove(f"Saves AIs/{name_ai}.json")
        except:
            pass

    def update(self, ai_name=""):
        """Обновляем сохранение"""

        self.delete(ai_name)
        self.save(ai_name)

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

        def normalize(self, x, min=0, max=1):
            # Нормализуем от 0 до 1
            result = x - np.min(x)
            if np.max(x) != 0:
                result = result / np.max(result)

            # От min до max
            result = result * (max - min) + min
            return result


        def ReLU(self, x, return_derivative=False):
            """Не действует ограничение value_range"""

            if return_derivative:
                return (x > 0)

            return (x > 0) * x

        def ReLU_2(self, x, return_derivative=False):
            if return_derivative:
                return (x < 0) * 0.01 + \
                       np.multiply(0 <= x, x <= 1) + \
                       (x > 1) * 0.01

            return (x < 0) * 0.01 * x + \
                   np.multiply(0 <= x, x <= 1) * x + \
                   (x > 1) * 0.01 * x

        def Softmax(self, x, return_derivative=False):
            return np.exp(x) / np.sum(np.exp(x))

        def Tanh(self, x, return_derivative=False):
            if return_derivative:
                return 1 / (10 * np.power(np.cosh(.1*x), 2))

            return np.tanh(.1*x)

        def Sigmoid(self, x, return_derivative=False):
            if return_derivative:
                return np.exp(-.1*x) / (10 * np.power(1 + np.exp(-.1*x), 2))

            return 1 / (1 + np.exp(-.1*x))
