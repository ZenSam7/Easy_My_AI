import numpy as np


class AI:
    """Набор функций для работы с самодельным ИИ"""
    def __init__(self):
        self.weights =      []          # Появиться после вызова create_weights

        self.activation_function = self.ActivationFunctions()
        self.what_activation_function = self.activation_function.ReLU_2  # Какую функцию активации используем
        self.end_activation_function  = self.activation_function.Tanh    # Какую функцию активации используем для выходных значений

        self.alpha = 1e-1     # Альфа каэффицент (каэффицент скорости обучения) (настраивается самостоятельно)
        self.have_bias_neuron = False      # Определяет наличие нейрона смещения (True или False)
        self.number_disabled_neurons = 0.0      # Какую долю нейронов "отключаем" при обучении

        self.packet_size = 1     # Как много ошибок будем усреднять, чтобы на основе этой усреднённой ошибки изменять веса
        # Чем packet_size больше, тем "качество обучения" меньше, но скорость итераций обучения больше
        self.packet_errors = []   # Где мы будем эти ошибки складывать

        self.gamma = 0
        self.epsilon = 0
        self.q_alpha = 0

        self.q = []
        self.actions = []
        self.states = []
        self.last_state = []


    def create_weights(self, architecture: list, add_bias_neuron=False, min_weight=-1, max_weight=1):
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
            print(result_layer_neurons.shape)
            print(layer_weight.shape)
            print()
            result_layer_neurons = self.what_activation_function(
                                        result_layer_neurons.dot(layer_weight) )


        # Добавляем ответ (единицу) для нейрона смещения, для последнего перемножения
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

        # "Разведуем окружающую среду"
        if self.epsilon < 1:    # Если адекватные значения epsilon, то ИСКАЖАЕМ ответ
            if np.random.random() > self.epsilon:   # С вероятностью epsilon
                ai_result *= np.random.randint(1, 10_000, ai_result.shape) /100     # Умножаем на число от 0.01 до 100

        else:    # А если неадекватные значения epsilon (>= 1), то заменяем ответ случайными числами
            ai_result = np.random.random(ai_result.shape)


        # Находим действие
        return self.actions[ np.argmax(ai_result) ]


    def learning(self, input_data: list, answer: list, get_error=False, squared_error=False):
        """Метод обратного распространения ошибки для изменения весов в нейронной сети"""

        # Нормализуем веса (очень грубо)
        if np.any(abs(self.weights[0]) >= 1e7):    # Если запредельные значения весов
            # То уменьшаем все их
            for i in range(len(self.weights)):  # На каждом слое
                while np.any(abs(self.weights[i]) >= 10):    # До адекватного состояния
                    self.weights[i] /= 10

            # И уменьшаем alpha
            self.alpha /= 10


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

        delta_weight = np.matrix(delta_weight)  # Превращаем вектор в матрицу


        self.packet_errors.append(np.sum(delta_weight))

        if self.packet_size == 1 or len(self.packet_errors) == self.packet_size:
            if self.packet_size != 1:
                # Замением пакет ошибок на их среднее
                delta_weight = np.mean(self.packet_errors)
                delta_weight = np.repeat(delta_weight, len(answer))


            for weight, layer_answer in zip(self.weights[::-1], answers[::-1]):
                # Превращаем вектор в матрицу
                delta_weight = np.matrix(delta_weight)
                layer_answer = np.matrix(layer_answer)


                # Матрица, предотвращающая переобучение, умножением изменением веса рандомных нейронов на 0
                dropout_mask = np.random.random(size=(delta_weight.shape[1], layer_answer.shape[1])) \
                               >= self.number_disabled_neurons

                # Изменяем веса
                weight -= ( np.multiply(dropout_mask, # Отключаем изменение некоторых связей
                                        self.alpha * delta_weight.T.dot(layer_answer)) ).T

                # "Переносим" на другой слой и умножаем на производную
                delta_weight = np.multiply( delta_weight.dot(weight.T),
                                            self.what_activation_function(layer_answer, True) )

                # К нейрону смещения не идут связи, поэтому обрезаем этот нейрон смещения
                if self.have_bias_neuron:
                    delta_weight = np.matrix(delta_weight.T.tolist()[0:-1]).T


            if get_error:
                if self.packet_size == 1:    # Без усреднения
                    err = np.sum( np.power(answer - ai_answer, 2) )   # Квадратичное отклонение
                    return err

                else:       # С усреднением
                    return np.mean(np.power(self.packet_errors, 2))


            self.packet_errors = []


    def make_all_for_q_learning(self, actions: list, gamma=0.5, epsilon=0.0, q_alpha=0.1):
        """Создаём всё необходимое для Q-обучения \n
        Q-таблицу (таблица вознаграждений за действие) \n
        Каэфицент важности будущего вознаграждения gamma \n
        Каэфицент почти случайных действий epsilon \n
        Каэфицент скорости изменения Q-таблицы q_alpha """

        self.actions = actions
        self.gamma = gamma      # Каэфицент на сколько важно будущее вознаграждение
        self.epsilon = epsilon  # Каэфицент "разведки окружающей среды"
        self.q_alpha = q_alpha

        self.q = [[0 for _ in range(len(actions))]]    # Таблица состояний (заполняем нулевым состоянием)

        # Заполняем "первое" (несуществующее (т.к. мы в прошлом на 1 шаг)) состояние количеством входов
        self.states = [[-0.0 for _ in range(self.weights[0].shape[0])]]
        self.last_state = self.states[0]     # Прошлое состояние нужно для откатывания на 1 состояние назад


    def q_learning(self, state, reward_for_state, num_function=1, learning_method=2.1,
                   squared_error=False, recce_mode=False):
        """ Глубокое Q-обучение (ИИ используется как предсказатель правильных действий)

        recce_mode - при значении True включаем "режим разведки", т.е. при таком режиме ИИ не обучается, а только пополняется Q-таблица

-------------------------- \n

        num_function - это номер функции обновления Q-таблицы: \n
        \n 1: Q(s,a) = Q(s,a) + α[r + γ(max Q(s’,a')) - Q(s,a)] \n
        \n 2: Q(s,a) = Q(s,a) + α[r + γ Q(s’,a') - Q(s,a)] \n
        \n 3: Q(s,a) = Q(s,a) + α[r + γ(Expected Q(s’,a')) - Q(s,a)] \n
        \n 4: Q(s,a) = R + γ Q’(s’,a’) \n
        \n 5: Q(s,a) = R + γ Q’(s’, max a) \n

-------------------------- \n

        Методы обучения (значение learning_method определяет) : \n
        1 : В качестве "правильного" ответа выбирается то, которое максимально вознаграждается, и на место действия \
        (которое приводит к лучшему ответу) ставиться максимальное значение функции активации \
        (self.activation_function.max), а на остальные места минимум функции активации (self.activation_function.min) \n
        P.s. Это неочень хорошо, т.к. игнорируются другие варианты, которые приносят либо столько же, либо немного меньше  \
        вознаграждения (а вибирается только один "правильный"). НО ОН ХОРОШО ПОДХОДИТ, КОГДА У ВАС В ЗАДАЧЕ ИМЕЕТСЯ ИСКЛЮЧИТЕЛЬНО 1 \
        ПРАВИЛЬНЫЙ ОТВЕТ, А "БОЛЕЕ" И "МЕНЕЕ" ПРАВИЛЬНЫХ БЫТЬ НЕ МОЖЕТ \n
        \n

        2 : Делаем ответы которые больше вознаграждаются, более "правильным" \n
        Дробная часть числа означает, в какую степень будем возводить "стремление у лучшим результатам" (что это такое читай в P.s.) \
        (чем степень больше, тем этот режим будет больше похож на режим 1. НАПРИМЕР: 2.2 означает, что мы используем метод обучения 2 и возводим в степень 2 \
        "стремление у лучшим результатам", а 2.345 означает, что степень будет равна 3.45 ) \n
        P.s. Работает так: Сначала переводим значения вознаграждений в промежуток от 0 до 1 (т.е. где вместо максимума вознаграждения\
        - 1, в место минимума - 0, а остальные вознаграждения между ними (без потерь "расстояний" между числами)) \
        потом прибавляем 0.5 и возводим в степень "стремление у лучшим результатам" (уже искажаем "расстояние" между числами) \
        (чтобы ИИ больше стремился именно к лучшим результатам и совсем немного учитывал остальные \
        (чем степень больше, тем меньше учитываются остальные результаты))  \n
        """

        STATE = self.states.index(self.last_state)


        # Q-обучение

        # Формируем "правильный" ответ
        if learning_method == 1:
            answer = [self.activation_function.min for _ in range(len(self.actions))]

            # На месте максимального значения из Q-таблицы ставим максимально возможное значение как "правильный" ответ
            answer[self.q[STATE].index( max(self.q[STATE]) )] =\
                    self.activation_function.max

        elif 2 < learning_method < 3:
            # Нам нужны значения от минимума функции активации до максимума функции активации
            # Переводим в промежуток от 0 до 1
            answer = np.array(self.q[STATE]) - min(self.q[STATE])
            if np.max(answer) != 0:
                answer = answer / np.max(answer)    # Не работает с /=

            # Искажаем "расстояние" между числами
            answer = answer + 0.5
            answer = np.power(answer, (learning_method -2) *10 )

            # Опять переводим в промежуток от 0 до 1
            answer = answer - np.min(answer)
            if np.max(answer) != 0:
                answer = answer / np.max(answer)    # Не работает с /=

            # Переводим в промежуток от min до max
            answer = answer * (self.activation_function.max - self.activation_function.min) +\
                    self.activation_function.min

            answer = answer.tolist()


        # Если режим разведки выключен
        if recce_mode == False:
            # Обновляем Q-таблицу
            self._update_q_table(state, reward_for_state, num_function)

            # Изменяем веса (не забываем, что мы находимся в состоянии на 1 шаг назад)
            self.learning(self.last_state, answer, squared_error=squared_error)

        else:
            # Иначе просто обновляем таблицу с изменёнными параметрами
            Gamma, Epsilon, Q_alpha = self.gamma, self.epsilon, self.q_alpha
            self.gamma, self.epsilon, self.q_alpha = 0.5, 1, 0.5

            self._update_q_table(state, reward_for_state, num_function)

            self.gamma, self.epsilon, self.q_alpha = Gamma, Epsilon, Q_alpha


    def _update_q_table(self, state, reward_for_state, num_function):
        """Формулы для обновления Q-таблицы \n

        --------------------------

        \n 1: Q(s,a) = Q(s,a) + α[r + γ(max Q(s’,a')) - Q(s,a)] \n
        \n 2: Q(s,a) = Q(s,a) + α[r + γ Q(s’,a') - Q(s,a)] \n
        \n 3: Q(s,a) = Q(s,a) + α[r + γ(Expected Q(s’,a')) - Q(s,a)] \n
        \n 4: Q(s,a) = R + γ Q’(s’,a’) \n
        \n 5: Q(s,a) = R + γ Q’(s’, max a) \n
        """

        # Если не находим состояние в прошлых состояниях (Q-таблице), то добовляем новое
        if not state in self.states:
            self.states.append(state)
            self.q.append([0 for _ in range(len(self.actions))])


        # Откатываем наше состояние на 1 шаг назад (Текущее (по реальному времени) == Будущее, а Прошлое == Настоящее)
        STATE = self.states.index(self.last_state)    # "Текущее" на самом деле прошлое
        FUTURE_STATE = self.states.index(state)       # "Будущее" на самом деле настоящее

        # С учётом вышенаписанного, наше "текущее" действие == ответ нейронки на прошлое состояние
        ACT     =    self.actions.index( self.q_start_work(self.last_state) )
        FUTURE_ACT = self.actions.index( self.q_start_work(state) )


        self.last_state = state      # А "прошлое" уже является настоящим (т.к. все переменные уже объявлены)


        if num_function == 1:
            self.q[STATE][ACT] = self.q[STATE][ACT] + self.q_alpha * \
                                 (reward_for_state + self.gamma * max(self.q[FUTURE_STATE]) - self.q[STATE][ACT])

        elif num_function == 2:
            self.q[STATE][ACT] = self.q[STATE][ACT] + self.q_alpha *\
                                 (reward_for_state + self.gamma * self.q[FUTURE_STATE][FUTURE_ACT] - self.q[STATE][ACT])

        elif num_function == 3:
            self.q[STATE][ACT] = self.q[STATE][ACT] +  self.q_alpha * \
                                 (reward_for_state +  self.gamma * sum(self.q[FUTURE_STATE]) - self.q[STATE][ACT])

        elif num_function == 4:
            self.q[STATE][ACT] = reward_for_state + self.gamma * self.q[FUTURE_STATE][FUTURE_ACT]

        elif num_function == 5:
            self.q[STATE][ACT] = reward_for_state + self.gamma * max(self.q[FUTURE_STATE])


    def save_data(self, name_this_ai: str):
        """Сохраняет всю необходимую информацию о текущей ИИ"""

        with open("Data of AIs.txt", "a+") as file:
            file.write("name " + name_this_ai + "\n")

            file.write("weights " +
                       "".join((str([i.tolist() for i in self.weights]).split()))
                       + "\n")
            file.write("what_activation_function " + str(self.what_activation_function) + "\n")
            file.write("end_activation_function " + str(self.end_activation_function) + "\n")
            file.write("alpha " + str(self.alpha) + "\n")
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
            file.write("last_state " +
                       "".join((str(self.last_state).split()))
                       + "\n")
            file.write("gamma " + str(self.gamma) + "\n")
            file.write("epsilon " + str(self.epsilon) + "\n")
            file.write("q_alpha " + str(self.q_alpha) + "\n")


            file.write("\n")


    def _find_among_data(self, start_with_ai_name: str, what_find: str, from_bottom_to_top=True):
        """Ищет среди сохранённых данных нужное нам слово (после имени), и возвращает его значение"""

        # Если читаем снизу вверх, то читаем инвертированный файл (шаг == -1)
        from_bottom_to_top = -1 if from_bottom_to_top else 1

        # Если мы читаем снизу вверх, то когда найдём имя, останется лишь найти нужную переменную в ЭТОМ списке
        reverlsed_file_list = []

        with open("Data of AIs.txt") as file:
            find_name = False

            for line in file.readlines()[::from_bottom_to_top]:
                if from_bottom_to_top == -1:
                    reverlsed_file_list.append(line)

                # Если нашли название, то записываем данные
                if not find_name and start_with_ai_name == line[5:-1]:
                    find_name = True

                if find_name:
                    if from_bottom_to_top == -1:
                        for LINE in reverlsed_file_list[::from_bottom_to_top]:
                            if what_find == LINE[:len(what_find)]:
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


    def load_data(self, AI_name: str):
        """Загружает все данные сохранённой ИИ"""
        """Она загружает последнее сохранение (последнее имя), если несколько одинаковых имён"""

        self.weights = [np.array(i) for i in self._find_among_data(AI_name, "weights", True)]
        self.alpha = self._find_among_data(AI_name, "alpha", True)
        self.have_bias_neuron = True if self._find_among_data(AI_name, "have_bias_neuron", True) == True else False
        self.number_disabled_neurons = self._find_among_data(AI_name, "number_disabled_neurons", True)
        self.packet_size = self._find_among_data(AI_name, "packet_size", True)
        self.activation_function.min = self._find_among_data(AI_name, "value_range", True)[0]
        self.activation_function.max = self._find_among_data(AI_name, "value_range", True)[1]
        self.q = self._find_among_data(AI_name, "q_table", True)
        self.states = self._find_among_data(AI_name, "states", True)
        self.actions = self._find_among_data(AI_name, "actions", True)
        self.q_alpha = self._find_among_data(AI_name, "q_alpha", True)
        self.epsilon = self._find_among_data(AI_name, "epsilon", True)
        self.last_state = self._find_among_data(AI_name, "last_state", True)


        # Выясняем какая функция активации

        result = self._find_among_data(AI_name, "what_activation_function", True).split()[2].split('.')[-1]
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
        result = self._find_among_data(AI_name, "end_activation_function", True).split()
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


    def delete_data(self, AI_name: str):
        """Удаляет последнее сохранение данный (если такое имя повторяется)"""

        # Копируем
        with open("Data of AIs.txt", "r+") as file:
            lines = file.readlines()
            file.truncate(0)


        # Удаляем последние данные
        for num in range(1, len(lines) +1):
            ind = len(lines) - num      # Снизу вверх
            line = lines[ind]

            if line[5:-1] == AI_name:
                for _ in range(17):
                    lines.pop(ind)
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

        def value_range(self, min, max):
            """Задаём область значений"""
            self.min = min
            self.max = max


        def ReLU(self, x, return_derivative=False):
            """Не действует ограничение value_range"""

            if return_derivative:
                return (x > 0) * 1

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
                return 0.05* (max-min) / np.power(np.cosh(0.1*x), 2)

            else:
                return 0.5* ( (max-min) *np.tanh(0.1*x) +min+max)

        def Sigmoid(self, x, return_derivative=False):
            min = self.min
            max = self.max

            if return_derivative:
                return - ( (min - max) * np.exp(-0.1*x) ) / (10* np.power(1 + np.exp(-0.1*x), 2))

            else:
                return ((max - min) / (1 + np.exp(-0.1*x))) + min
