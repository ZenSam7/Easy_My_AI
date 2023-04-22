import numpy as np


class AI:
    """Набор функций для работы с самодельным ИИ"""
    def __init__(self):
        self.weights =      []      # Появиться после вызова create_weights
        self.architecture = []      # Появиться после вызова create_weights
        self.alpha =        0.001    # Альфа каэффицент (каэффицент скорости обучения)
        self.activation_function = self.ActivationFunctions()
        self.what_activation_function = self.activation_function.Sigmoid # Какую функцию активации используем
        self.end_activation_function = None     # Какую функцию активации используем для выходных зачений
        self.have_bias_neuron = 0    # Определяет наличие нейрона смещения (True or False)


    def create_weights(self, architecture: list, add_bias_neuron=False):
        """Создаёт матрицу со всеми весами между всеми элементами
        (Подавать надо список с количеством нейронов на каждом слое (архитектуру нейрпонки))
        ((Все веса - рандомные числа от -1 до 1))"""

        self.architecture = architecture  # Добавляем архитектуру (что бы было)

        self.have_bias_neuron = add_bias_neuron


        def create_weight(inp, outp):
            """Создаём веса между inp и outp"""
            layer_weights = []

            for _ in range(inp):
                layer_weights.append([])  # Этот список заполним весами
                for _ in range(outp):
                    # Добавляем дробь от -1 до 1
                    layer_weights[-1].append( np.random.randint(-100, 100) / 100)
            return np.array(layer_weights)

        # Добавляем все веса между слоями нейронов
        for i in range(len(architecture) -1):
            self.weights.append(create_weight(architecture[i] + add_bias_neuron,
                                              architecture[i + 1]))



    def start_work(self, input_data: list, return_answers=False):
        """Возвращает результат работы нейронки, из входных данных"""

        # Определяем входные данные как вектор
        result_layer_neurons = np.array(input_data)

        # Сохраняем список всех ответов от нейронов каждого слоя
        list_answers = [result_layer_neurons]


        # Проходимся по каждому (кроме последнего) слою весов
        for layer_weight in self.weights[:-1]:
            # Если есть нейрон смещения, то в вправо result_layer_neurons добавляем еденицы
            # Чтобы можно было умножить еденицы на веса нейрона смещения
            if self.have_bias_neuron:
                result_layer_neurons = np.array(result_layer_neurons.tolist() + [1])

            # Процежеваем через функцию активации  ...
            # ... Результат перемножения результата прошлого слоя на слой весов
            result_layer_neurons = self.what_activation_function(
                                        result_layer_neurons.dot(layer_weight) )


            if return_answers:
                list_answers.append(result_layer_neurons)



        if self.have_bias_neuron:
            result_layer_neurons = np.array(result_layer_neurons.tolist() + [1])

        # Пропускаем выходные данные через последнюю функцию активации (Если есть)
        if self.end_activation_function == None:
            result_layer_neurons = result_layer_neurons.dot(self.weights[-1])
        else:
            result_layer_neurons = self.end_activation_function(
                        result_layer_neurons.dot(self.weights[-1]))

        # Если нажо, возвращаем спосок с ответами от каждого слоя
        if return_answers:
            return [result_layer_neurons, list_answers]
        else:
            return result_layer_neurons



    def learning(self, input_data: list, answer: list, get_info=False):
        """Метод обратного распространения ошибки для изменения весов в нейронной сети"""

        # Определяем наш ответ как вектор
        answer = np.array(answer)
        # Определяем наши входные данные как вектор
        input_data = np.array(input_data)

        # То, что выдала нам нейросеть
        ai_answer = self.start_work(input_data)
        # Список с ответами от каждого слоя нейронов
        answers_ai = self.start_work(input_data, True)[1]


        error = np.sum((answer - ai_answer) **2 )

        # На сколько должны суммарно изменить веса
        delta_w = answer - ai_answer


        # Матрица, предотвращающая переобучение, умножением рандомных нейронов на 0
        #dropout_mask = np.random.randint(2, size=layer_l.shape)


        for weight, layer_answer in zip(self.weights[::-1], answers_ai[::-1]):
            # Обратно распространяем, то, на сколько надо исправить суммарно все веса (учитываем веса)
            delta_w = delta_w.dot(weight.T)
            # Умножаем производную функции активации на сколько надо изменить веса
            delta_w *= self.what_activation_function(layer_answer, True)

            # Изменяем веса
            weight -= self.alpha * layer_answer.T.dot(delta_w)



    def save_data(self, name_this_ai: str):
        """Сохраняет все переменные текущей ИИ"""

        with open("Data of AIs.txt", "a+") as file:
            file.write("name " + name_this_ai + "\n")

            file.write("weights " +
                       "".join((str([i.tolist() for i in self.weights]).split()))
                       + "\n")
            file.write("architecture " +
                       "".join((str(self.architecture).split()))
                       + "\n")
            file.write("alpha " + str(self.alpha) + "\n")
            file.write("what_activation_function " + str(self.what_activation_function) + "\n")
            file.write("end_activation_function " + str(self.end_activation_function) + "\n")
            file.write("have_bias_neuron " + str(self.have_bias_neuron) + "\n")


            file.write("\n")
        print("Все данные сохранены ✔")


    def find_among_data(self, start_with_ai_name: str, what_find: str, from_bottom_to_top=False):
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
                                    from json import loads
                                    return loads(value)
                                elif value[0].isdigit(): # Либо число
                                    return float(value)
                                else:                    # Либо название
                                    return str(value)

                    else:
                        if what_find == line[0:len(what_find)]:
                            value = line[len(what_find) + 1:-1]
                            if value[0] == "[":      # Либо список
                                from json import loads
                                return loads(value)
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

        self.weights = [np.array(i) for i in self.find_among_data(load_AI_with_name, "weights", True)]
        self.architecture = self.find_among_data(load_AI_with_name, "architecture", True)
        self.alpha = self.find_among_data(load_AI_with_name, "alpha", True)
        self.have_bias_neuron = self.find_among_data(load_AI_with_name, "have_bias_neuron", True)

        # Выясняем какая функция активации
        result = self.find_among_data(load_AI_with_name, "what_activation_function", True).split()[1]
        if result == "AI.activation_function.ReLU":
            self.what_activation_function = self.activation_function.ReLU
        elif result == "AI.activation_function.ReLU_2":
            self.what_activation_function = self.activation_function.ReLU_2
        elif result == "AI.activation_function.Gaussian":
            self.what_activation_function = self.activation_function.Gaussian
        elif result == "AI.activation_function.SoftPlus":
            self.what_activation_function = self.activation_function.SoftPlus
        elif result == "AI.activation_function.Curved":
            self.what_activation_function = self.activation_function.Curved
        elif result == "AI.activation_function.Tanh":
            self.what_activation_function = self.activation_function.Tanh
        elif result == "AI.activation_function.Sigmoid":
            self.what_activation_function = self.activation_function.Sigmoid

        # То же самое для end_activation_function
        result = self.find_among_data(load_AI_with_name, "end_activation_function", True).split()[1]
        if result == "None":
            self.end_activation_function = None
        elif result == "AI.activation_function.ReLU":
            self.end_activation_function = self.activation_function.ReLU
        elif result == "AI.activation_function.ReLU_2":
            self.end_activation_function = self.activation_function.ReLU_2
        elif result == "AI.activation_function.Gaussian":
            self.end_activation_function = self.activation_function.Gaussian
        elif result == "AI.activation_function.SoftPlus":
            self.end_activation_function = self.activation_function.SoftPlus
        elif result == "AI.activation_function.Curved":
            self.end_activation_function = self.activation_function.Curved
        elif result == "AI.activation_function.Tanh":
            self.end_activation_function = self.activation_function.Tanh
        elif result == "AI.activation_function.Sigmoid":
            self.end_activation_function = self.activation_function.Sigmoid





    class ActivationFunctions:
        """Набор функций активации и их производных"""

        def __init__(self):
            self.min = -1
            self.max = 1

        def value_range(self, min: int, max: int):
            """Задаём область значений"""
            self.min = min
            self.max = max


        def ReLU(self, x, return_derivative=False):
            """ReLU"""

            if return_derivative:
                if x < 0:
                    return 0.1
                else:
                    return 1

            else:
                if x < 0:
                    return 0.1 * x
                else:
                    return x

        def ReLU_2(self, x, return_derivative=False):
            """Таже ReLU, но немного другая"""

            if return_derivative:
                if x < 0:
                    return 0.1
                elif x <= 1:  # 0 <= x <= 1
                    return 1
                else:
                    return 0.1

            else:
                if x < 0:
                    return 0.1 * x
                elif x <= 1:    # 0 <= x <= 1
                    return x
                else:
                    return 0.1 * x +0.9

        def Gaussian(self, x, return_derivative=False):
            """Распределение Гаусса"""
            min = self.min
            max = self.max

            if return_derivative:
                return -.2 * (max - min) * x * np.exp(-.1* x**2)

            else:
                return (max - min) * np.exp(-.1* x**2 ) + min

        def SoftPlus(self, x, return_derivative=False):
            """ 'Типа экспонента' """
            min = self.min

            if return_derivative:
                return np.exp(x) / (np.exp(x) + 1)

            else:
                return np.log(1 + np.exp(x)) + min

        def Curved(self, x, return_derivative=False):
            """Как ReLU, только плавная"""

            if return_derivative:
                return x / (np.sqrt(2 * x ** 2 + 1)) + 1

            else:
                return ( np.sqrt(2* x**2 +1) -1 )/2 +x

        def Tanh(self, x, return_derivative=False):
            """Это Tanh (точно)"""
            min = self.min
            max = self.max

            if return_derivative:
                return (2* (max-min) * np.exp(2*x)) / ( (( np.exp(2*x) +1))**2)

            else:
                return (min-max) / ( np.exp(2*x) +1) + max

        def Sigmoid(self, x, return_derivative=False):
            """Cигмоида"""
            min = self.min
            max = self.max

            if return_derivative:
                return ( - (min - max) * np.exp(-0.1*x) ) / (10* (1 + np.exp(-0.1*x)) ** 2)

            else:
                return ((max - min) / (1 + np.exp(-0.1*x))) + min
