import numpy as np


class AI:
    """Набор функций для работы с самодельным ИИ"""
    def __init__(self):
        self.weights =      []          # Появиться после вызова create_weights

        self.activation_function = self.ActivationFunctions()
        self.what_activation_function = self.activation_function.ReLU_2  # Какую функцию активации используем
        self.end_activation_function  = self.activation_function.ReLU_2  # Какую функцию активации используем для выходных зачений

        self.alpha = 1e-7     # Альфа каэффицент (каэффицент скорости обучения)
        self.have_bias_neuron = False      # Определяет наличие нейрона смещения (True or False)
        self.number_disabled_neurons = 0.0      # Какую долю нейронов "отключаем" при обучении

        self.packet_size = 1     # Как много ошибок будем усреднять, чтобы на основе этой усреднённой ошибки изменять веса
        # Чем packet_size больше, тем "качество обучения" меньше, но скорость итераций обучения больше
        self.packet_errors = []   # Где мы будем эти ошибки складывать

        self.q = []         # Q-table
        self.states = []    # Все состояния
        self.actions = []     # Все действия
        self.gamma = 1
        self.epsilon = None
        self.q_alpha = 0.01



    def create_weights(self, architecture: list, add_bias_neuron=False,
                       min_weight=-1, max_weight=1):
        """Создаёт матрицу со всеми весами между всеми элементами
        (Подавать надо список с количеством нейронов на каждом слое (архитектуру нейронки))"""

        self.have_bias_neuron = add_bias_neuron

        # Добавляем все веса между слоями нейронов
        for i in range(len(architecture) -1):
            self.weights.append( np.random.randint(min_weight * 1000, max_weight * 1000,
                                                   size = (architecture[i] + add_bias_neuron,
                                                   architecture[i + 1])) /1000)


    def genetic_crossing_with(self, ai):
        """Перемешевает веса между ЭТОЙ нейронкой и нейронкой В АРГУМЕНТЕ \n
            P.s. Не обязательно, чтобы количество связей (размеры матриц весов) были обинаковы"""

        for layer1, layer2 in zip(self.weights, ai.weights):
            for _ in range(layer1.shape[0] * layer1.shape[1]): # Для каждого элемента...
                if np.random.random() < 0.5:  # ... С шансом 50% ...
                    # ... Производим замену на вес из другой матрицы
                    layer1[np.random.randint(layer1.shape[0]), np.random.randint(layer1.shape[1])] =\
                        layer2[np.random.randint(layer2.shape[0]), np.random.randint(layer2.shape[1])]


    def get_mutations(self, mutation=0.01):
        """Создаёт рандомные веса в нейронке"""

        for layer in self.weights:                            # Для каждого слоя
            for _ in range(layer.shape[0] * layer.shape[1]):  # Для каждого элемента
                if np.random.random() <= mutation:            # С шансом mutation
                    # Производим замену на случайное число
                    layer[np.random.randint(layer.shape[0]), np.random.randint(layer.shape[1])] = \
                        np.random.random() - np.random.random()


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
            result_layer_neurons = self.what_activation_function(
                                        result_layer_neurons.dot(layer_weight) )


        # Добавляем ответ (единицу) для нейрона смещения
        if self.have_bias_neuron:
            result_layer_neurons = np.array(result_layer_neurons.tolist() + [1])
        if return_answers:
            list_answers.append(result_layer_neurons)


        # Пропускаем выходные данные через последнюю функцию активации (Если есть)
        if self.end_activation_function == None:
            result_layer_neurons = result_layer_neurons.dot(self.weights[-1])
        else:
            result_layer_neurons = self.end_activation_function(
                        result_layer_neurons.dot(self.weights[-1]))


        # Если надо, возвращаем спосок с ответами от каждого слоя
        if return_answers:
            return result_layer_neurons, list_answers
        else:
            return result_layer_neurons


    def q_start_work(self, input_data: list):
        """Возвращает action, на основе входных данных"""

        ai_result = self.start_work(input_data)

        # "Разведуем окружающую среду" ()
        if (self.epsilon != None) and (np.random.random() < self.epsilon):
            ai_result *= np.random.random(ai_result.shape) +0.5        # Умножаем на число от 0.5 до 1.5

        ai_result = ai_result.tolist()

        # Находим действие
        for i in range(len(ai_result)):
            if max(ai_result) == ai_result[i]:
                return self.actions[i]

        return self.actions[0]


    def learning(self, input_data: list, answer: list, get_error=False, squared_error=False):
        """Метод обратного распространения ошибки для изменения весов в нейронной сети"""

        # Определяем наш ответ как вектор
        answer = np.array(answer)
        # Определяем наши входные данные как вектор
        input_data = np.array(input_data)

        # То, что выдала нам нейросеть | Список с ответами от каждого слоя нейронов
        ai_answer, answers = self.start_work(input_data, True)


        # На сколько должны суммарно изменить веса
        if squared_error:
            delta_weight = np.power(ai_answer - answer, 2) *\
                           (-1* ((ai_answer - answer) <0) + 1*((ai_answer - answer) >=0) )
        else:
            delta_weight = ai_answer - answer


        self.packet_errors.append(np.sum(delta_weight))

        if self.packet_size == 1 or len(self.packet_errors) == self.packet_size:
            if self.packet_size != 1:       # Замением пакет ошибок на их среднее
                delta_weight = np.mean(self.packet_errors)
                delta_weight = np.repeat(delta_weight,  self.weights[-1].shape[1])
            self.packet_errors = []


            for weight, layer_answer in zip(self.weights[::-1], answers[::-1]):
                # Превращаем векторы в матрицу
                layer_answer = np.matrix(layer_answer)
                delta_weight = np.matrix(delta_weight)


                # Матрица, предотвращающая переобучение, умножением изменением веса рандомных нейронов на 0
                dropout_mask = np.random.random(size=(layer_answer.shape[1], delta_weight.shape[1])) \
                               >= self.number_disabled_neurons


                # Изменяем веса
                weight -= np.multiply(dropout_mask, # Отключаем изменение некоторых связей
                                      self.alpha * layer_answer.T.dot(delta_weight))


                # К нейрону смещения не идут связи, поэтому обрезаем этот нейрон смещения
                if self.have_bias_neuron:
                    weight = weight[0:-1]
                    layer_answer = np.matrix(layer_answer.tolist()[0][0:-1])

                delta_weight = delta_weight.dot(weight.T)
                delta_weight.dot( self.what_activation_function(layer_answer, True).T )


            if get_error:
                err = np.sum( np.power(answer - ai_answer, 2) )   # Квадратичное отклонение

                if not (np.isnan(err) or err == None or err is None or err == np.nan):
                    return err


    def make_all_for_q_learning(self, actions: list, gamma=0.1, epsilon=0.1, q_alpha=0.01):
        """Создаём всё необходимое для Q-обучения
            (q-таблицу, каэфицент вознаграждения gamma)"""

        self.actions = actions
        self.gamma = gamma     # Каэфицент "доверия опыту"
        self.epsilon = epsilon  # Каэфицент "разведки окружающей среды"
        self.q_alpha = q_alpha

        self.q = []    # Таблица состояний


    def q_learning(self, state, reward_for_state, future_state, num_function=2):
        """Q-обучение \n
        ИИ используется как предсказатель правильныз действий

        -----------------

        num_function - это номер функции обновления Q-таблицы: \n
        -----------------
        \n 1: Q(s,a) = Q(s,a) + α[r + γ(max Q(s’,a')) - Q(s,a)] \n
        \n 2: Q(s,a) = (1 - α) Q(s,a) + α[r + γ(max Q(s’,a))] \n
        \n 3: Q(s,a) = Q(s,a) + α[r + γ Q(s’,a') - Q(s,a)] \n
        \n 4: Q(s,a) = Q(s,a) + α[r + γ(Expected Q(s’,a')) - Q(s,a)] \n
        \n 5: Q(s,a) = R + γ Q’(s’,a’) \n
        \n 6: Q(s,a) = R + γ Q’(s’, max a) \n
        """

        action = self.q_start_work(state)

        state = [i for i in state]
        future_state = [i for i in future_state]


        # Если не находим состояние в прошлых состояниях, то добовляем новое
        if not state in self.states:
            self.states.append(state)
            self.q.append([0 for _ in range(len(self.actions))])
        if not future_state in self.states:
            self.states.append(future_state)
            self.q.append([0 for _ in range(len(self.actions))])


        STATE = self.states.index(state)
        ACT = self.actions.index(action)


        answer = [self.activation_function.min for _ in range(len(self.actions))]
        # На месте максимального значения из Q-таблицы ставим максимально возможное значение как "правильный" ответ
        answer[self.q[STATE].index( max(self.q[STATE]) )] =\
                self.activation_function.max

        # Изменяем веса
        self.learning(state, answer)


        # Обновляею Q-таблицу
        if num_function == 1:
            self.q[STATE][ACT] = self.q[STATE][ACT] +\
                                    self.q_alpha * (reward_for_state + \
                                                  self.gamma * max( self.q[self.states.index(future_state)] ) - \
                                                  self.q[STATE][ACT] )

        elif num_function == 2:
            self.q[STATE][ACT] = (1 - self.q_alpha) * self.q[STATE][ACT] + \
                                    self.q_alpha * (reward_for_state + \
                                                    self.gamma * max( self.q[self.states.index(future_state)] ) )

        elif num_function == 3:
            self.q[STATE][ACT] = self.q[STATE][ACT] + self.q_alpha * (reward_for_state + \
                                self.gamma * self.q[self.states.index(future_state)][self.actions.index(self.q_start_work(state))] -\
                                self.q[STATE][ACT])

        elif num_function == 4:
            self.q[STATE][ACT] = self.q[STATE][ACT] + \
                                 self.q_alpha * (reward_for_state + \
                                                 self.gamma * sum( self.q[self.states.index(future_state)] ) -\
                                                 self.q[STATE][ACT])

        elif num_function == 5:
            self.q[STATE][ACT] = reward_for_state + self.gamma * self.q[self.states.index(future_state)][self.actions.index(self.q_start_work(state))]

        elif num_function == 6:
            self.q[STATE][ACT] = reward_for_state + self.gamma * max( self.q[self.states.index(future_state)] )


    def save_data(self, name_this_ai: str):
        """Сохраняет все переменные текущей ИИ"""

        with open("Data of AIs.txt", "a+") as file:
            file.write("name " + name_this_ai + "\n")

            file.write("weights " +
                       "".join((str([i.tolist() for i in self.weights]).split()))
                       + "\n")
            file.write("what_activation_function " + str(self.what_activation_function) + "\n")
            file.write("end_activation_function " + str(self.end_activation_function) + "\n")
            file.write("alpha " + str(self.alpha) + "\n")
            file.write("q_alpha " + str(self.q_alpha) + "\n")
            file.write("have_bias_neuron " + str(self.have_bias_neuron) + "\n")
            file.write("number_disabled_neurons " + str(self.number_disabled_neurons) + "\n")
            file.write("packet_size " + str(self.packet_size) + "\n")
            file.write("value_range " + "".join((str([self.activation_function.min, self.activation_function.max]).split())) + "\n")
            file.write("q_table " +
                       "".join((str(self.q).split()))
                       + "\n")
            file.write("states " +
                       "".join((str(self.states).split()))
                       + "\n")
            file.write("actions " +
                       "".join((str(self.actions).split()))
                       + "\n")
            file.write("gamma " + str(self.gamma) + "\n")
            file.write("epsilon " + str(self.epsilon) + "\n")


            file.write("\n")


    def _find_among_data(self, start_with_ai_name: str, what_find: str, from_bottom_to_top=True):
        """Ищет среди сохранённых данных нужное нам слово, и возвращает его значение"""

        # Если читаем снизу вверх, то читаем инвертированный файл (шаг == -1)
        from_bottom_to_top = -1 if from_bottom_to_top else 1

        # Если мы читаем снизу вверх, то когда найдём имя, останется лишь найти нужную переменную в ЭТОМ списке
        reverlsed_file_list = []

        with open("Data of AIs.txt") as file:
            find_name = False

            for line in file.readlines()[::from_bottom_to_top]:
                # Если нашли название, то записываем данные

                if not find_name and start_with_ai_name == line[5 : 5 + len(start_with_ai_name)]:
                    find_name = True

                if find_name:
                    if from_bottom_to_top == -1:
                        for LINE in reverlsed_file_list:
                            if what_find == LINE[0:len(what_find)]:
                                value = LINE[len(what_find) + 1:-1]
                                if value[0] == "[":      # Либо список
                                    from ast import literal_eval
                                    return literal_eval(value)
                                elif value[0].isdigit(): # Либо число
                                    return float(value)
                                else:                    # Либо название
                                    return str(value)

                    else:
                        if what_find == line[0:len(what_find)]:
                            value = line[len(what_find) + 1:-1]
                            if value[0] == "[":      # Либо список
                                from ast import literal_eval
                                return literal_eval(value)
                            elif value[0].isdigit(): # Либо число
                                return float(value)
                            else:                    # Либо название
                                return str(value)


                # См. выше описание reverlsed_file_list
                if from_bottom_to_top == -1:
                    reverlsed_file_list.append(line)


    def load_data(self, load_AI_with_name: str):
        """Загружает все данные сохранённой ИИ"""
        """!! ВНИМАНИЕ !! Она загружает ПОСЛЕДНЕЕ сохранение (ПОСЛЕДНЕЕ имя), если несколько одинаковых имён"""

        self.weights = [np.array(i) for i in self._find_among_data(load_AI_with_name, "weights", True)]
        self.alpha = self._find_among_data(load_AI_with_name, "alpha", True)
        self.have_bias_neuron = True if self._find_among_data(load_AI_with_name, "have_bias_neuron", True) == True else False
        self.number_disabled_neurons = self._find_among_data(load_AI_with_name, "number_disabled_neurons", True)
        self.packet_size = self._find_among_data(load_AI_with_name, "packet_size", True)
        self.activation_function.min = self._find_among_data(load_AI_with_name, "value_range", True)[0]
        self.activation_function.max = self._find_among_data(load_AI_with_name, "value_range", True)[1]
        self.q = self._find_among_data(load_AI_with_name, "q_table", True)
        self.states = self._find_among_data(load_AI_with_name, "states", True)
        self.actions = self._find_among_data(load_AI_with_name, "actions", True)
        self.q_alpha = self._find_among_data(load_AI_with_name, "q_alpha", True)


        # Выясняем какая функция активации

        result = self._find_among_data(load_AI_with_name, "what_activation_function", True).split()[2].split('.')[-1]
        if result == "ReLU":
            self.what_activation_function = self.activation_function.ReLU
        elif result == "ReLU_2":
            self.what_activation_function = self.activation_function.ReLU_2
        elif result == "Gaussian":
            self.what_activation_function = self.activation_function.Gaussian
        elif result == "SoftPlus":
            self.what_activation_function = self.activation_function.SoftPlus
        elif result == "Curved":
            self.what_activation_function = self.activation_function.Curved
        elif result == "Tanh":
            self.what_activation_function = self.activation_function.Tanh
        elif result == "Sigmoid":
            self.what_activation_function = self.activation_function.Sigmoid

        # То же самое для end_activation_function
        result = self._find_among_data(load_AI_with_name, "end_activation_function", True).split()
        if result[0] != "None":
            result = result[2].split('.')[-1]

        if result == "None":
            self.end_activation_function = None
        elif result == "ReLU":
            self.end_activation_function = self.activation_function.ReLU
        elif result == "ReLU_2":
            self.end_activation_function = self.activation_function.ReLU_2
        elif result == "Gaussian":
            self.end_activation_function = self.activation_function.Gaussian
        elif result == "SoftPlus":
            self.end_activation_function = self.activation_function.SoftPlus
        elif result == "Curved":
            self.end_activation_function = self.activation_function.Curved
        elif result == "Tanh":
            self.end_activation_function = self.activation_function.Tanh
        elif result == "Sigmoid":
            self.end_activation_function = self.activation_function.Sigmoid


    def delete_data(self, load_AI_with_name: str):
        """Удаляет ПОСЛЕДНЕЕ сохранение данный (если такое имя повторяется)"""

        # Копируем
        with open("Data of AIs.txt", "r+") as file:
            lines = file.readlines()
            file.truncate(0)


        # Удаляем последние данные
        for num in range(1, len(lines)):
            line = lines[len(lines) - num] # Снизу вверх

            if line[5:-1] == load_AI_with_name:
                for _ in range(16):
                    lines.pop(len(lines) - num +1)
                break


        # Записываем обратно
        with open("Data of AIs.txt", "r+") as file:
            for line in lines:
                file.write(line)



    class ActivationFunctions:
        """Набор функций активации и их производных"""

        def __init__(self):
            self.min = 0
            self.max = 1

        def value_range(self, min: int, max: int):
            """Задаём область значений"""
            self.min = min
            self.max = max


        def ReLU(self, x, return_derivative=False):
            """Не действует ограничение value_range"""

            if return_derivative:
                return (x > 0)

            else:
                return (x > 0) * x

        def ReLU_2(self, x, return_derivative=False):
            min = self.min
            max = self.max

            if return_derivative:
                return (x < min) * 0.01 + \
                       np.multiply(min <= x, x <= max) + \
                       (x > max) * 0.01

            else:
                return (x < min) * 0.01 * x + \
                       np.multiply(min <= x, x <= max) * x + \
                       (x > max) * 0.01 * x

        def Curved(self, x, return_derivative=False):
            """Не действует ограничение value_range"""

            if return_derivative:
                return x / (np.sqrt(2 * np.power(x,2) + 1)) + 1

            else:
                return ( np.sqrt(2* np.power(x,2) +1) -1 )/2 +x

        def SoftPlus(self, x, return_derivative=False):
            """Не действует ограничение value_range"""
            min = self.min

            if return_derivative:
                return np.exp(x) / (np.exp(x) + 1)

            else:
                return np.log(1 + np.exp(x)) + min

        def Softmax(self, x, return_derivative=False):
            """Не действует ограничение value_range"""

            if return_derivative:
                return np.exp(x) / np.sum(np.exp(x))

            else:
                return np.exp(x) / np.sum(np.exp(x))

        def Gaussian(self, x, return_derivative=False):
            min = self.min
            max = self.max

            if return_derivative:
                return -.2 * (max - min) * x * np.exp(-.1* np.power(x,2))

            else:
                return (max - min) * np.exp(-.1* np.power(x,2) ) + min

        def Tanh(self, x, return_derivative=False):
            min = self.min
            max = self.max

            if return_derivative:
                return 0.05* (max-min) / np.power(cosh(.1*x), 2)

            else:
                return 0.5* ( (max-min) *np.tanh(.1*x) +min+max)



        def Sigmoid(self, x, return_derivative=False):
            min = self.min
            max = self.max

            if return_derivative:
                return - ( (min - max) * np.exp(-0.1*x) ) / (10* np.power(1 + np.exp(-0.1*x), 2))

            else:
                return ((max - min) / (1 + np.exp(-0.1*x))) + min

