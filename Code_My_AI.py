import numpy as np
import json
from enum import Enum
from typing import Callable, List


class ImpossibleContinueLearning(Exception):
    pass


class AI:
    """Набор функций для работы с самодельным ИИ"""
    def __init__(self):
        self.name = str(np.random.randint(2**31))
        self.weights = []          # Появиться после вызова create_weights

        self.kit_act_func = ActivationFunctions
        self.kit_upd_q_table = FuncsUpdateQTable
        # Какую функцию активации используем
        self.what_act_func = self.kit_act_func.Tanh
        # Какую функцию активации используем для выходных значений
        self.end_act_func  = self.kit_act_func.Tanh

        self.alpha = 1e-2     # Альфа каэффицент (каэффицент скорости обучения)
        self.have_bias_neuron = True      # Определяет наличие нейрона смещения
        self.number_disabled_weights = 0.0      # Какую долю весов "отключаем" при обучении

        # Чем больше, тем "качество" обучения больше (но до определённого момента)
        # (Не влияет на скорость обучения)
        self.batch_size = 1
        self.packet_delta_weight = []
        self.packet_layer_answers = []

        self.q_table = {}     # Q-table
        # self.states = []    # Все состояния
        self.actions = []     # Все действия
        self.recce_mode = False
        self.gamma = 0.1
        self.epsilon = 0
        self.q_alpha = 0.1


    def create_weights(self, architecture: List[int], add_bias_neuron=True,
                       min_weight=-1, max_weight=1):
        """Создаёт матрицу со всеми весами между всеми элементами
        (Подавать надо список с количеством нейронов на каждом слое (архитектуру нейронки))"""

        self.have_bias_neuron = add_bias_neuron
        self.architecture = architecture
        self.weights = []

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


    def start_work(self, input_data: List[int], return_answers=False) -> np.ndarray:
        """Возвращает результат работы нейронки, из входных данных"""
        # Определяем входные данные как вектор
        result_layer_neurons = np.array(input_data)

        # Сохраняем список всех ответов от нейронов каждого слоя
        list_answers = []

        try:
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

        except ValueError:
            print("Проверьте размерность входных/выходных данных и "
                  "количество входных/выходных нейронов у ИИшки")
            raise ImpossibleContinueLearning

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


    def learning(self, input_data: List[int], answer: List[int],
                 get_error=False,
                 squared_error=False):
        """Метод обратного распространения ошибки для изменения весов в нейронной сети \n"""

        # Определяем наш ответ как вектор
        answer = np.array(answer)

        # То, что выдала нам нейросеть | Список с ответами от каждого слоя нейронов
        ai_answer, answers = self.start_work(input_data, True)

        # Нормализуем веса (очень грубо)
        if np.any(np.abs(self.weights[0]) >= 1e6):    # Если запредельные значения весов
            # То пересоздаём веса
            self.create_weights(self.architecture, self.have_bias_neuron)
            # И уменьшаем alpha
            self.alpha /= 10

        # На сколько должны суммарно изменить веса
        delta_weight = ai_answer - answer
        if squared_error:
            delta_weight = np.power(ai_answer - answer, 2) * \
                           (-1* ((ai_answer - answer) <0) + 1*((ai_answer - answer) >=0) )
                           # Оставляем знак ↑


        # Реализуем batch_size
        if len(self.packet_delta_weight) != self.batch_size:
            self.packet_delta_weight.append(delta_weight)
            self.packet_layer_answers.append(answers)
            return

        else:
            # Когда набрали нужное количество кладываем все данные
            delta_weight = np.sum(self.packet_delta_weight, axis=0)

            # Складываем ответы от каждого слоя из пакета
            summ_answers = [np.array(ans) for ans in self.packet_layer_answers[0]]

            for layer_index in range(len(self.packet_layer_answers[0])):
                for list_answers in self.packet_layer_answers[1:]: # Первые ответы уже в summ_answers
                    summ_answers[layer_index] += np.array(list_answers[layer_index])

            answers = summ_answers

            self.packet_delta_weight.clear()
            self.packet_layer_answers.clear()

        # Совершаем всю магию здесь
        for weight, layer_answer in zip(self.weights[::-1], answers[::-1]):
            # Превращаем векторы в матрицу
            layer_answer = np.matrix(layer_answer)
            delta_weight = np.matrix(delta_weight)

            gradient = layer_answer.T.dot(delta_weight)

            # Матрица, предотвращающая переобучение
            # Умножаем изменение веса рандомных нейронов на 0
            if self.number_disabled_weights > 0:
                dropout_mask = np.random.random(size=(layer_answer.shape[1], delta_weight.shape[1])) \
                               >= self.number_disabled_weights

                gradient = np.multiply(dropout_mask, # Отключаем изменение некоторых связей
                                        layer_answer.T.dot(delta_weight))

            # Изменяем веса
            weight -= self.alpha * gradient

            # К нейрону смещения не идут связи, поэтому обрезаем этот нейрон смещения
            if self.have_bias_neuron:
                weight = weight[0:-1]
                layer_answer = np.matrix(layer_answer.tolist()[0][0:-1])

            # "Переносим" градиент на другой слой (+ умножаем на производную)
            delta_weight = delta_weight.dot(weight.T)
            delta_weight.dot(self.what_act_func(layer_answer, True).T)


    def q_start_work(self, input_data: List[int], return_index=False) -> str:
        """Возвращает action, на основе входных данных"""
        ai_result = self.start_work(input_data).tolist()

        # "Разведуем окружающую среду" (берём случайное действие)
        if (self.epsilon != 0) and (np.random.random() < self.epsilon):
            if return_index:
                return np.random.randint(len(self.actions))
            return self.actions[np.random.randint(len(self.actions))]

        # Находим действие
        if return_index:
            return np.argmax(ai_result)
        return self.actions[np.argmax(ai_result)]


    def make_all_for_q_learning(self, actions: List[str],
                func_update_q_table: Callable,
                gamma=0.3, epsilon=0.0, q_alpha=0.1):
        """Создаём всё необходимое для Q-обучения \n
        Q-таблицу (таблица вознаграждений за действие), каэфицент вознаграждения gamma, \
        каэфицент почти случайных действий epsilon, и каэфицент скорости изменения Q-таблицы q_alpha

        --------------------------

        func_update_q_table - это функция обновления Q-таблицы (выбирается из kit_upd_q_table) \n
        \n standart:   Q(s,a) = Q(s,a) + α[r + γ(max Q(s’,a')) - Q(s,a)] \n
        \n future:     Q(s,a) = Q(s,a) + α[r + γ Q(s’,a') - Q(s,a)] \n
        \n future_sum: Q(s,a) = Q(s,a) + α[r + γ(Expected Q(s’,a')) - Q(s,a)] \n
        \n simple:     Q(s,a) = R + γ Q’(s’,a’) \n
        \n simple_max: Q(s,a) = R + γ Q’(s’, max a) \n

        \n"""

        self.actions = actions
        self.gamma = gamma      # Каэфицент "доверия опыту"
        self.epsilon = epsilon  # Каэфицент "разведки окружающей среды"
        self.q_alpha = q_alpha

        self.q_table = {}    # Хэш-Таблица состояний
        self.func_update_q_table = func_update_q_table


    def q_learning(self, state: list, reward_for_state: float, future_state: list,
                   learning_method=2.2,
                   squared_error=False,
                   recce_mode=False,
                   update_q_table=True,
                   ):
        """
        ИИ используется как предсказатель правильных действий\n

        update_q_table - обновлять Q-таблицу
        add_new_states - добавлять новые состояния в Q-таблицу
        (СОВЕТУЮ ПОСТАВИТЬ recce_mode=True, А ПОТОМ add_new_states=False;
        Т.К. МОЖНО БУДЕТ ОЧЕНЬ СИЛЬНО СОКРАТИТЬ ВРЕМЯ НА ОБУЧЕНИЕ)

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

        # Добовляем новые состояния в Q-таблицу
        state_str = str(state)
        future_state_str = str(future_state)

        default = [0] * len(self.actions)
        self.q_table.setdefault(state_str, default)
        self.q_table.setdefault(future_state_str, default)


        if recce_mode:
            Epsilon, self.epsilon = self.epsilon, 2
            self._update_q_table(state, reward_for_state, future_state)
            self.epsilon = Epsilon
            return

        # Q-обучение

        # Формируем "правильный" ответ
        if learning_method == 1:
            answer = [0 for _ in range(len(self.actions))]

            # На месте максимального значения из Q-таблицы ставим
            # максимально возможное значение как "правильный" ответ
            answer[self.q_table[state_str].index(max(self.q_table[state_str]))] = 1

        elif 2 < learning_method < 3:
            # Нам нужны значения от минимума функции активации до максимума функции активации
            answer = self.kit_act_func.normalize(np.array(self.q_table[state_str]))

            # Искажаем "расстояние" между числами
            answer = answer + 0.5
            answer = np.power(answer, (learning_method -2) *10 )

            answer = self.kit_act_func.normalize(answer).tolist()

        else:
            raise "Неверный метод обучения"

        # Изменяем веса
        if self.learning(state, answer, squared_error=squared_error):
            return

        # Обновляем Q-таблицу
        if (update_q_table or add_new_states):
            self._update_q_table(state, reward_for_state, future_state)


    def _update_q_table(self, state: list, reward_for_state: float, future_state: int):
        state_str = str(state)
        future_state_str = str(future_state)

        act = self.q_start_work(state, True)

        all_kwargs = {
            "q_table": self.q_table,
            "q_start_work": self.q_start_work,
            "q_alpha": self.q_alpha,
            "reward_for_state": reward_for_state,
            "gamma": self.gamma,
            "state": state,
            "state_str": state_str,
            "future_state": future_state,
            "future_state_str": future_state_str,
            "act": act,
        }

        self.q_table[state_str][act] = self.func_update_q_table(**all_kwargs)


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

        def get_name_upd_func(func):
            func_str = str(func)

            if "standart" in func_str:
                return "standart"
            elif "future" in func_str:
                return "future"
            elif "future_sum" in func_str:
                return "future_sum"
            elif "simple" in func_str:
                return "simple"
            elif "simple_max" in func_str:
                return "simple_max"

        ai_data = {}

        ai_data["weights"] = [i.tolist() for i in self.weights]

        ai_data["q_table"] = self.q_table
        # ai_data["states"] = self.states
        # ai_data["last_state"] = self.last_state
        ai_data["actions"] = self.actions

        ai_data["architecture"] = self.architecture
        ai_data["have_bias_neuron"] = self.have_bias_neuron

        ai_data["number_disabled_neurons"] = self.number_disabled_weights
        ai_data["alpha"] = self.alpha
        ai_data["batch_size"] = self.batch_size

        ai_data["what_activation_function"] = get_name_act_func(self.what_act_func)
        ai_data["end_activation_function"] = get_name_act_func(self.end_act_func)
        ai_data["func_update_q_table"] = get_name_upd_func(self.func_update_q_table)

        # ai_data["last_reward"] = 0
        ai_data["gamma"] = self.gamma
        ai_data["epsilon"] = self.epsilon
        ai_data["q_alpha"] = self.q_alpha

        # Сохраняем ИИшку ЛЮБОЙ ценой
        try:
            with open(f"Saves AIs/{name_ai}.json", "w+") as save_file:
                json.dump(ai_data, save_file)
        except BaseException as e:
            with open(f"Saves AIs/{name_ai}.json", "w+") as save_file:
                json.dump(ai_data, save_file)
            raise e

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

        def get_upd_func_with_name(name):
            if name == "standart":
                return self.kit_upd_q_table.standart
            elif name == "future":
                return self.kit_upd_q_table.future
            elif name == "future_sum":
                return self.kit_upd_q_table.future_sum
            elif name == "simple":
                return self.kit_upd_q_table.simple
            elif name == "simple_max":
                return self.kit_upd_q_table.simple_max

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
            self.func_update_q_table = get_upd_func_with_name(ai_data["func_update_q_table"])

            self.q_table = ai_data["q_table"]

            self.states = ai_data["states"]
            # self.last_state = ai_data["last_state"]
            self.actions = ai_data["actions"]

            # self.last_reward = ai_data["last_reward"]
            self.gamma = ai_data["gamma"]
            self.epsilon = ai_data["epsilon"]
            self.q_alpha = ai_data["q_alpha"]

        except FileNotFoundError:
            print(f"Сохранение {name_ai} не найдено")

    def delete(self, ai_name=""):
        """Удаляет сохранение"""

        name_ai = self.name if ai_name == "" else ai_name

        try:
            os.remove(f"Saves AIs/{name_ai}.json")
        except :
            pass

    def update(self, ai_name=""):
        """Обновляем сохранение"""

        self.delete(ai_name)
        self.save(ai_name)


    def print_how_many_parameters(self):
        parameters = []
        for layer in self.weights:
            parameters.append(layer.shape[0] * layer.shape[1])

        print(f"{self.name} \t\t",
              f"Parameters: {sum(parameters)} \t\t",
              f" {self.architecture} ", end="")

        if self.have_bias_neuron:
            print("+ bias", end="")

        print()


class FuncsUpdateQTable(Enum):
    """Формулы для обновления Q-таблицы: \n

    \n standart:   Q(s,a) = Q(s,a) + α[r + γ(max Q(s’,a')) - Q(s,a)] \n
    \n future:     Q(s,a) = Q(s,a) + α[r + γ Q(s’,a') - Q(s,a)] \n
    \n future_sum: Q(s,a) = Q(s,a) + α[r + γ(Expected Q(s’,a')) - Q(s,a)] \n
    \n simple:     Q(s,a) = R + γ Q’(s’,a’) \n
    \n simple_max: Q(s,a) = R + γ Q’(s’, max a) \n
    """

    @staticmethod
    def standart(q_table={}, state_str=None, act=0, future_state_str=None,
                 q_alpha=0.0, reward_for_state=0, gamma=0.0, **kwargs):
        return q_table[state_str][act] + \
            q_alpha * (reward_for_state + gamma * \
                       max(q_table[future_state_str]) - q_table[state_str][act])

    @staticmethod
    def future(q_table={}, state_str=None, act=0, future_state=None, future_state_str=None,
               q_alpha=0.0, reward_for_state=0, gamma=0.0,
               q_start_work=None, **kwargs):
        return q_table[state_str][act] + \
            q_alpha * (reward_for_state + gamma * \
                       q_table[future_state_str][q_start_work(future_state, True)] - q_table[state_str][act])

    @staticmethod
    def future_sum(q_table={}, state_str=None, act=0, future_state_str=None,
                   q_alpha=0.0, reward_for_state=0, gamma=0.0, **kwargs):
        return q_table[state_str][act] + q_alpha * \
            (reward_for_state + gamma * sum(q_table[future_state_str]) - q_table[state_str][act])

    @staticmethod
    def simple(q_table={}, future_state_str=None, future_state=None,
               reward_for_state=0, gamma=0.0,
               q_start_work=None, **kwargs):
        return reward_for_state + \
            gamma * q_table[future_state_str][q_start_work(future_state, True)]

    @staticmethod
    def simple_max(q_table={}, future_state_str=None,
                   reward_for_state=0, gamma=0.0, **kwargs):
        return reward_for_state + gamma * max(q_table[future_state_str])

class ActivationFunctions:
    """Набор функций активации и их производных"""

    @staticmethod
    def normalize(x: np.ndarray, min=0, max=1):
        # Нормализуем от 0 до 1
        result = x - np.min(x)
        if np.max(x) != 0:
            result = result / np.max(result)

        # От min до max
        result = result * (max - min) + min
        return result

    @staticmethod
    def ReLU(x: np.ndarray, return_derivative=False):
        """Не действует ограничение value_range"""

        if return_derivative:
            return (x > 0)

        return (x > 0) * x

    @staticmethod
    def ReLU_2(x: np.ndarray, return_derivative=False):
        if return_derivative:
            return (x < 0) * 0.01 + \
                np.multiply(0 <= x, x <= 1) + \
                (x > 1) * 0.01

        return (x < 0) * 0.01 * x + \
            np.multiply(0 <= x, x <= 1) * x + \
            (x > 1) * 0.01 * x

    @staticmethod
    def Softmax(x: np.ndarray, return_derivative=False):
        return np.exp(x) / np.sum(np.exp(x))

    @staticmethod
    def Tanh(x: np.ndarray, return_derivative=False):
        if return_derivative:
            return 1 / (10 * np.power(np.cosh(.1 * x), 2))

        return np.tanh(.1 * x)

    @staticmethod
    def Sigmoid(x: np.ndarray, return_derivative=False):
        if return_derivative:
            return np.exp(-.1 * x) / (10 * np.power(1 + np.exp(-.1 * x), 2))

        return 1 / (1 + np.exp(-.1 * x))
