import numpy as np


class AI:
    """Набор функций для работы с самодельным ИИ"""

    def __init__(self):
        self.weights = []          # Появиться после вызова create_weights
        self.architecture = []     # Появиться после вызова create_weights
        self.moments = 0       # Сохраняем значение предыдущего импульса
        self.velocitys = 0     # Сохраняем значение предыдущей скорости

        self.act_func = self.ActivationFunctions()
        self.what_act_func = self.act_func.ReLU_2  # Какую функцию активации используем
        self.end_act_func  = self.act_func.Tanh    # Какую функцию активации используем для выходных значений

        self.alpha = 1e-1     # Альфа коэффициент (коэффициент скорости обучения) (настраивается самостоятельно)
        self.have_bias_neuron = False      # Определяет наличие нейрона смещения (True или False)
        self.number_disabled_weights = 0.0      # Какую долю нейронов "отключаем" при обучении

        self.batch_size = 1     # Как много ошибок будем усреднять, чтобы на основе этой усреднённой ошибки изменять веса
        # Чем batch_size больше, тем "качество обучения" меньше, но скорость итераций обучения больше
        self.packet_errors = []   # Где мы будем эти ошибки складывать
        self.type_error = 1

        self.gamma = 0
        self.epsilon = 0
        self.q_alpha = 0
        self.recce_mode = False

        self.q = []
        self.actions = []
        self.states = []
        self.last_state = []


    def create_weights(self, architecture: list, add_bias_neuron=True, min_weight=-1, max_weight=1):
        """Создаёт матрицу со всеми весами между всеми элементами
        (Подавать надо список с количеством нейронов на каждом слое (архитектуру нейронки))"""

        self.have_bias_neuron = add_bias_neuron
        self.architecture = architecture
        self.weights = []

        # Добавляем все веса между слоями нейронов
        for i in range(len(architecture) -1):
            self.weights.append(
                self.act_func.normalize(
                    np.random.random(size = (architecture[i] + add_bias_neuron,
                                            architecture[i + 1])),
                    min_weight, max_weight
                )
            )

        self.moments = [0 for _ in range(len(architecture))]
        self.velocitys = [0 for _ in range(len(architecture))]


    def genetic_crossing_with(self, ai):
        """Перемешивает веса между ЭТОЙ нейронкой и нейронкой В АРГУМЕНТЕ \n
            P.s. Не обязательно, чтобы количество связей (размеры матриц весов) были одинаковы"""

        for layer1, layer2 in zip(self.weights, ai.weights):
            for _ in range(layer1.shape[0] * layer1.shape[1]): # Для каждого элемента...
                if np.random.random() < 0.5:  # ... С шансом 50% ...
                    # ... Производим замену на вес из другой матрицы
                    layer1[np.random.randint(layer1.shape[0]), np.random.randint(layer1.shape[1])] =\
                        layer2[np.random.randint(layer2.shape[0]), np.random.randint(layer2.shape[1])]


    def get_mutations(self, mutation=0.05):
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
                                        result_layer_neurons.dot(
                                            layer_weight * \
                                            (np.random.random(size=layer_weight.shape) >= self.number_disabled_weights) * \
                                            (1 + self.number_disabled_weights) )
                                        )


        # Добавляем ответ для нейрона смещения (единицу), для последнего перемножения (для end_act_func)
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


        # Если надо, возвращаем список с ответами от каждого слоя
        if return_answers:
            return result_layer_neurons, list_answers
        else:
            return result_layer_neurons


    def q_start_work(self, input_data: list):
        """Возвращает action, на основе входных данных"""

        ai_result = self.start_work(input_data)

        # "Разведуем окружающую среду"
        if np.random.random() < self.epsilon or self.recce_mode:   # С вероятностью epsilon
            ai_result = np.random.random(ai_result.shape)

        # Находим действие
        return self.actions[ np.argmax(ai_result) ]


    def learning(self, input_data: list, answer: list,
                 get_error=False, type_error=1,
                 type_regularization=1, regularization_value=2, regularization_coefficient=0.1,
                 impulse_coefficient=0.9):
        """Метод обратного распространения ошибки для изменения весов в нейронной сети \n
        Ошибки могут быть: \n
        1: (regular:) |ai_answer - answer| / len(answer) \n
        2: (quadratic:) (ai_answer - answer)^2 / len(answer) \n
        3: (logarithmic:) ln^2( (ai_answer - answer) +1 ) / len(answer) \n
    ------------------ \n

        regularization: \n
        1: delta += SameSign * reg_coeff * sqrt( sum( (weights/regular_val) ^2) ) \n
        2: delta += SameSign * reg_coeff * sum( abs( weights * (abs(weights) >= regular_val) ) -regular_val ) \n
    ------------------ \n

        regularization_value: In what interval (±) do we keep weights \n
    ------------------ \n

        regularization_coefficient: How hard the AI will try to keep the weights \n
    ------------------ \n

        impulse_coefficient: 0 < x < 1, This factor affects "remembering" the direction in which the gradient \
            is moving (the direction the AI weights move towards the minimum error) (usually around 0.9)"""


        # Нормализуем веса (очень грубо)
        if np.any([ np.any( abs(i) >= 1e6 ) for i in self.weights ]):    # Если запредельные значения весов
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
        if type_error == 2:
            delta_weight = np.power(delta_weight, 2) * \
                           (-1 * (delta_weight < 0) + 1 * (delta_weight >= 0)) # Тут сохраняем знак
        elif type_error == 3:
            delta_weight = np.power( np.log(ai_answer-answer +1), 2) * \
                           (-1 * (delta_weight < 0) + 1 * (delta_weight >= 0)) # Тут сохраняем знак


        # Регуляризация (держим веса близкими к 0, чтобы не улетали в космос)
        if type_regularization == 1:
            delta_weight += (-1 * (delta_weight < 0) + 1 * (delta_weight >= 0)) * \
                        regularization_coefficient *\
                        np.sqrt( np.sum([ np.sum(np.power(i/regularization_value, 2)) for i in self.weights ]) )
        elif type_regularization == 2:
            delta_weight += (-1 * (delta_weight < 0) + 1 * (delta_weight >= 0)) * \
                        regularization_coefficient * \
                        np.sum([np.sum(np.abs( i * (np.abs(i) >= regularization_value) ) - regularization_value) for i in self.weights])

        delta_weight = np.matrix(delta_weight)  # Превращаем вектор в матрицу



        self.packet_errors.append(delta_weight)

        if len(self.packet_errors) == self.batch_size:
            if self.batch_size != 1:
                # Заменим пакет ошибок на средний вектор ошибки
                sum_delta = 0   # (пустой вектор)
                for delta in self.packet_errors:
                    sum_delta += delta

                delta_weight = sum_delta / self.batch_size

            for weight, layer_answer, moment, velocity in \
                    zip(self.weights[::-1], answers[::-1], self.moments[::-1], self.velocitys[::-1]):
                # Превращаем вектор в матрицу
                delta_weight = np.matrix(delta_weight)
                layer_answer = np.matrix(layer_answer)


                # Изменяем веса с оптимизацией Adam
                gradient = ( delta_weight.T.dot(layer_answer) ).T
                moment  =  impulse_coefficient * moment  +  (1 -impulse_coefficient) * gradient
                velocity = impulse_coefficient * velocity + (1 -impulse_coefficient) * np.power(gradient, 2)

                MOMENT = moment / (1 -impulse_coefficient)
                VELOCITY = velocity / (1 -impulse_coefficient)

                weight -= self.alpha * ( MOMENT / (np.sqrt(VELOCITY) + 1 ) )


                # "Переносим" на другой слой и умножаем на производную
                delta_weight = np.multiply(delta_weight.dot(weight.T),
                                           self.what_act_func(layer_answer, True))

                # К нейрону смещения не идут связи, поэтому обрезаем этот нейрон смещения
                if self.have_bias_neuron == True:
                    delta_weight = np.matrix(delta_weight.T.tolist()[0:-1]).T


            if get_error:
                if self.batch_size == 1:    # Без усреднения
                    err = np.sum( np.abs(self.packet_errors) ) / answer.shape[0]
                    self.packet_errors.clear()
                    return err

                else:       # С усреднением
                    err = np.mean([np.abs(i) for i in self.packet_errors]) / answer.shape[0]
                    self.packet_errors.clear()
                    return err

            else:
                self.packet_errors.clear()


    def make_all_for_q_learning(self, actions: list, gamma=0.5, epsilon=0.0, q_alpha=0.1):
        """Создаём всё необходимое для Q-обучения \n
        Q-таблицу (таблица вознаграждений за действие) \n
        Коэффициент важности будущего вознаграждения gamma \n
        Коэффициент почти случайных действий epsilon \n
        Коэффициент скорости изменения Q-таблицы q_alpha """

        self.actions = actions
        self.gamma = gamma      # Коэффициент на сколько важно будущее вознаграждение
        self.epsilon = epsilon  # Коэффициент "разведки окружающей среды"
        self.q_alpha = q_alpha

        # Таблица состояний (заполняем нулевым состоянием)
        # Размеры: States ⨉ Actions
        # # States — количество уникальных состояний (С каждым новым состояние пополняется)
        self.q = [[0 for _ in range(len(actions))]]

        # Заполняем "первое" (несуществующее (т.к. мы в прошлом на 1 шаг)) состояние количеством входов
        self.states = [[-0.0 for _ in range(self.weights[0].shape[0] - self.have_bias_neuron)]]
        self.last_state = self.states[0]     # Прошлое состояние нужно для откатывания на 1 состояние назад
        self.last_reward = 0


    def q_learning(self, state, reward_for_state,
                   num_update_function=1, learning_method=2.1,
                   type_error=1, recce_mode=False,
                   type_regularization=1, regularization_value=2, regularization_coefficient=0.1,
                   impulse_coefficient=0.9):
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
        1: regular: |ai_answer - answer| \n
        2: quadratic: (ai_answer - answer)^2 \n
        3: logarithmic: ln^2( (ai_answer - answer) +1 ) \n

-------------------------- \n

        recce_mode - при значении True включаем "режим разведки", т.е. при таком режиме ИИ не обучается, а только пополняется Q-таблица

-------------------------- \n

        Методы обучения (значение learning_method определяет) : \n
        1 : В качестве "правильного" ответа выбирается то, которое максимально вознаграждается, и на место действия \
        (которое приводит к лучшему ответу) ставиться максимальное значение функции активации \
        (self.act_func.max), а на остальные места минимум функции активации (self.act_func.min) \n
        P.s. Это неочень хорошо, т.к. игнорируются другие варианты, которые приносят либо столько же, либо немного меньше  \
        вознаграждения (а выбирается только один "правильный"). НО ОН ХОРОШО ПОДХОДИТ, КОГДА У ВАС В ЗАДАЧЕ ИМЕЕТСЯ ИСКЛЮЧИТЕЛЬНО 1 \
        ПРАВИЛЬНЫЙ ОТВЕТ, А "БОЛЕЕ" И "МЕНЕЕ" ПРАВИЛЬНЫХ БЫТЬ НЕ МОЖЕТ \n
        \n

        2 : Делаем ответы которые больше вознаграждаются, более "правильным" \n
        Дробная часть числа означает, в какую степень будем возводить "стремление у лучшим результатам" (что это такое читай в P.s.) \
        (чем степень больше, тем этот режим будет больше похож на режим 1. НАПРИМЕР: 2.2 означает, что мы \
         используем метод обучения 2 и возводим в степень 2 "стремление у лучшим результатам", а 2.345 означает, что степень будет равна 3.45 ) \n
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
            answer = [self.act_func.min for _ in range(len(self.actions))]

            # На месте максимального значения из Q-таблицы ставим максимально возможное значение как "правильный" ответ
            answer[self.q[STATE].index( max(self.q[STATE]) )] =\
                    self.act_func.max

        elif 2 < learning_method < 3:
            # Искажаем "расстояние" между числами
            answer = self.act_func.normalize(np.array(self.q[STATE]), 0, 2)
            answer = np.power(answer, (learning_method -2) *10 )

            # Переводим в промежуток от min до max
            answer = self.act_func.normalize(np.array(self.q[STATE]), self.act_func.min, self.act_func.max)

            answer = answer.tolist()


        # Если режим разведки выключен
        if self.recce_mode == False:
            # Обновляем Q-таблицу
            self._update_q_table(state, reward_for_state, num_update_function)

            # Изменяем веса (не забываем, что мы находимся в состоянии на 1 шаг назад)
            self.learning(self.last_state, answer, type_error=type_error,
                          type_regularization=type_regularization, regularization_value=regularization_value,
                          regularization_coefficient=regularization_coefficient,
                          impulse_coefficient=impulse_coefficient)

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

        # Если не находим состояние в прошлых состояниях (Q-таблице), то добавляем новое
        if not state in self.states:
            self.states.append(state)
            self.q.append([0 for _ in range(len(self.actions))])


        # Откатываем наше состояние на 1 шаг назад (Текущее (по реальному времени) == Будущее, а Прошлое == Настоящее)
        REWARD = self.last_reward
        STATE = self.states.index(self.last_state)    # "Текущее" на самом деле прошлое
        FUTURE_STATE = self.states.index(state)       # "Будущее" на самом деле настоящее

        # С учётом выше написанного, наше "текущее" действие == ответ нейронки на прошлое состояние
        if self.epsilon < 1:
            ACT = self.actions.index( self.q_start_work(self.last_state) )
            FUTURE_ACT = self.actions.index( self.q_start_work(state) )
        else:
            ACT = np.random.randint(len(self.actions))        # Случайное действие
            FUTURE_ACT = np.random.randint(len(self.actions)) # Случайное действие

        # А "прошлое" уже является настоящим (т.к. все переменные уже объявлены)
        self.last_state = state
        self.last_reward = reward_for_state


        if num_update_function == 1:
            self.q[STATE][ACT] = self.q[STATE][ACT] + self.q_alpha * \
                                 (REWARD + self.gamma * max(self.q[FUTURE_STATE]) - self.q[STATE][ACT])

        elif num_update_function == 2:
            self.q[STATE][ACT] = self.q[STATE][ACT] + self.q_alpha *\
                                 (REWARD + self.gamma * self.q[FUTURE_STATE][FUTURE_ACT] - self.q[STATE][ACT])

        elif num_update_function == 3:
            self.q[STATE][ACT] = self.q[STATE][ACT] +  self.q_alpha * \
                                 (REWARD + self.gamma * sum(self.q[FUTURE_STATE]) - self.q[STATE][ACT])

        elif num_update_function == 4:
            self.q[STATE][ACT] = REWARD + self.gamma * self.q[FUTURE_STATE][FUTURE_ACT]

        elif num_update_function == 5:
            self.q[STATE][ACT] = REWARD + self.gamma * max(self.q[FUTURE_STATE])


    def save(self, name_this_ai: str):
        """Сохраняет всю необходимую информацию о текущей ИИ"""

        with open("Data of AIs.txt", "a+") as file:
            file.write("name " + name_this_ai + "\n")

            file.write("weights " +
                       "".join((str([i.tolist() for i in self.weights]).split()))
                       + "\n")
            file.write("what_act_func " + str(self.what_act_func) + "\n")
            file.write("end_act_func " + str(self.end_act_func) + "\n")
            file.write("alpha " + str(self.alpha) + "\n")
            file.write("have_bias_neuron " + str(self.have_bias_neuron) + "\n")
            file.write("number_disabled_weights " + str(self.number_disabled_weights) + "\n")
            file.write("architecture " +
                       "".join((str(self.architecture).split()))
                       + "\n")
            file.write("batch_size " + str(self.batch_size) + "\n")
            file.write("value_range " + "".join((str([self.act_func.min, self.act_func.max]).split())) + "\n")
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
            file.write("last_reward " + str(0) + "\n")
            file.write("gamma " + str(self.gamma) + "\n")
            file.write("epsilon " + str(self.epsilon) + "\n")
            file.write("q_alpha " + str(self.q_alpha) + "\n")


            file.write("\n")


    def _find_among_data(self, start_with_ai_name: str, what_find: str, from_bottom_to_top=True):
        """Ищет среди сохранённых данных нужное нам слово (после имени), и возвращает его значение"""

        # Если читаем снизу вверх, то читаем инвертированный файл (шаг == -1)
        from_bottom_to_top = -1 if from_bottom_to_top else 1

        # Если мы читаем снизу вверх, то когда найдём имя, останется лишь найти нужную переменную в ЭТОМ списке
        reversed_file_list = []

        with open("Data of AIs.txt") as file:
            find_name = False

            for line in file.readlines()[::from_bottom_to_top]:
                if from_bottom_to_top == -1:
                    reversed_file_list.append(line)

                # Если нашли название, то записываем данные
                if not find_name and start_with_ai_name == line[5:-1]:
                    find_name = True

                if find_name:
                    if from_bottom_to_top == -1:
                        for LINE in reversed_file_list[::from_bottom_to_top]:
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


    def load(self, AI_name: str):
        """Загружает все данные сохранённой ИИ"""
        """Она загружает последнее сохранение (последнее имя), если несколько одинаковых имён"""

        self.weights = [np.array(i) for i in self._find_among_data(AI_name, "weights", True)]
        self.architecture = self._find_among_data(AI_name, "architecture", True)
        self.alpha = float(self._find_among_data(AI_name, "alpha", True))
        self.have_bias_neuron = True if self._find_among_data(AI_name, "have_bias_neuron", True) == "True" else False
        self.number_disabled_weights = float(self._find_among_data(AI_name, "number_disabled_weights", True))
        self.batch_size = self._find_among_data(AI_name, "batch_size", True)
        self.act_func.min = self._find_among_data(AI_name, "value_range", True)[0]
        self.act_func.max = self._find_among_data(AI_name, "value_range", True)[1]
        self.q = self._find_among_data(AI_name, "q_table", True)
        self.states = self._find_among_data(AI_name, "states", True)
        self.actions = self._find_among_data(AI_name, "actions", True)
        self.q_alpha = self._find_among_data(AI_name, "q_alpha", True)
        self.epsilon = float(self._find_among_data(AI_name, "epsilon", True))
        self.last_state = self._find_among_data(AI_name, "last_state", True)
        self.last_reward = float(self._find_among_data(AI_name, "last_reward", True))


        # Выясняем какая функция активации

        result = self._find_among_data(AI_name, "what_act_func", True).split()[2].split('.')[-1]
        if result == "ReLU":
            self.what_act_func = self.act_func.ReLU
        elif result == "ReLU_2":
            self.what_act_func = self.act_func.ReLU_2
        elif result == "Gaussian":
            self.what_act_func = self.act_func.Gaussian
        elif result == "SoftPlus":
            self.what_act_func = self.act_func.SoftPlus
        elif result == "Curved":
            self.what_act_func = self.act_func.Curved
        elif result == "Tanh":
            self.what_act_func = self.act_func.Tanh
        elif result == "Sigmoid":
            self.what_act_func = self.act_func.Sigmoid

        # То же самое для end_act_func
        result = self._find_among_data(AI_name, "end_act_func", True).split()
        if result[0] != "None":
            result = result[2].split('.')[-1]

        if result == "None":
            self.end_act_func = None
        elif result == "ReLU":
            self.end_act_func = self.act_func.ReLU
        elif result == "ReLU_2":
            self.end_act_func = self.act_func.ReLU_2
        elif result == "Gaussian":
            self.end_act_func = self.act_func.Gaussian
        elif result == "SoftPlus":
            self.end_act_func = self.act_func.SoftPlus
        elif result == "Curved":
            self.end_act_func = self.act_func.Curved
        elif result == "Tanh":
            self.end_act_func = self.act_func.Tanh
        elif result == "Sigmoid":
            self.end_act_func = self.act_func.Sigmoid


    def delete(self, AI_name: str):
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
                for _ in range(19):
                    lines.pop(ind)
                break


        # Записываем обратно
        with open("Data of AIs.txt", "r+") as file:
            for line in lines:
                file.write(line)


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
                       np.multiply(min <= x, x <= max) *1 + \
                       (x > max) * 0.01

            else:
                return (x < min) * 0.01 * x + \
                       np.multiply(min <= x, x <= max) * x + \
                       ((x > max) * 0.01 * x +0.99)

        def Softmax(self, x, return_derivative=False):
            return np.exp(x) / np.sum(np.exp(x))

        def Tanh(self, x, return_derivative=False):
            min = self.min
            max = self.max

            if return_derivative:
                return 0.5* (max-min) / (np.power(np.cosh(x), 2))

            else:
                return 0.5* ( (max-min) *np.tanh(x) +min+max )

        def Sigmoid(self, x, return_derivative=False):
            min = self.min
            max = self.max

            if return_derivative:
                return ( (max- min) * np.exp(-1*x) ) / (np.power(1 + np.exp(-1*x), 2))

            else:
                return (max - min) / (1 + np.exp(-1*x)) + min
