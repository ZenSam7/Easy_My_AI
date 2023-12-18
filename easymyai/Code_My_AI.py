import numpy as np
import json
from os import remove, listdir, mkdir
from typing import Callable, List, Dict, Tuple, Optional

from .Ai_Funcs import *


class ImpossibleContinue(Exception):
    pass


class MyProperties(object):
    @classmethod
    def get_propertry(cls, property_func: Callable, attr_name: str, doc: str) -> property:
        """Создаём объект property со свойством property_func (которое мы выбираем из этого же класса)"""

        return property(fget=cls.property_getter(attr_name),
                        fset=cls.property_setter(property_func, attr_name),
                        fdel=cls.property_deleter,
                        doc=doc, )

    @classmethod
    def property_getter(cls, attr_name: str) -> float:
        """Возвращаем атрибут из класса AI (замыкание нужно чтобы мы знали какой атрибут
         мы хотим получить, при этом не вызывая функцию)"""
        name = attr_name

        def getter(cls) -> float:
            nonlocal name

            return cls.__dict__["_AI__" + name]

        return getter

    def property_setter(proiperty_func: Callable, attr_name: str) -> Callable:
        """По аналогии с property_getter, но при этом мы не достаём, а записываем в класс AI наш атрибут"""

        def property(cls, value: float):
            # Проверяем, подходит ли под выбранное свойство
            proiperty_func(value)

            # Если к нам попал AI_with_ensemble, то для него у каждой ИИшки
            # устанавливаем значение коэффициента
            if "ais" in cls.__dict__:
                for ai in cls.__dict__["ais"]:
                    ai.__dict__["_AI__" + attr_name] = value

            # Если просто работаем с AI
            else:
                # Если ошибки не произошло, то перезаписываем атрибут
                cls.__dict__["_AI__" + attr_name] = value

        return property

    def property_deleter(*args):
        print(f"Слышь, пёс, {args} нельзя удалять! Они вообще-то используются!!!")

    @staticmethod
    def from_1_to_0(value: float) -> Exception:
        """Свойство"""
        if not 0 <= value < 1:
            raise ImpossibleContinue(f"Число {value} не в диапазоне [0; 1)")

    @staticmethod
    def non_negative(value: float) -> Exception:
        """Свойство"""
        if value < 0:
            raise ImpossibleContinue(f"Число {value} меньше 0")

    @staticmethod
    def only_uint(value: float) -> Exception:
        """Свойство"""
        if not isinstance(value, int) or value < 0:
            raise ImpossibleContinue(f"Число {value} не целое и/или отрицательное")


class AI:
    """Набор функций для работы с самодельным ИИ"""

    # В чём соль, все коэффициенты для ИИшки я сделал с определёнными свойствами, чтобы пользователь не мог
    # как-то неправильно их изменить (в моей библиотеке от этого ничего не сломается, но ИИшке сразу поплохеет)
    # При этом, внутри библиотеке пользуемся секретными коэффициентами (__alpha, __epsilon, __gamma ...), но
    # пользователю даём просто alpha, epsilon, gamma ...

    # Стандартные коэффициенты
    alpha: float = MyProperties.get_propertry(
        MyProperties.from_1_to_0, "alpha",
        "Коэффициент скорости обучения")
    batch_size: int = MyProperties.get_propertry(
        MyProperties.only_uint, "batch_size",
        "Сколько входных данный усредняем при обучении")
    number_disabled_weights: float = MyProperties.get_propertry(
        MyProperties.from_1_to_0, "number_disabled_weights",
        "Какую долю весов \"отключаем\" при обучении")

    impulse1: float = MyProperties.get_propertry(
        MyProperties.from_1_to_0, "impulse1",
        "Коэффициент импульса (для оптимизатора Adam)")
    impulse2: float = MyProperties.get_propertry(
        MyProperties.from_1_to_0, "impulse2",
        "Коэффициент импульса (для оптимизатора Adam)")

    l1: float = MyProperties.get_propertry(
        MyProperties.from_1_to_0, "l1",
        "Коэффициент регуляризатора L1")
    l2: float = MyProperties.get_propertry(
        MyProperties.from_1_to_0, "l2",
        "Коэффициент регуляризатора L2")

    # Для Q-обучения
    epsilon: float = MyProperties.get_propertry(
        MyProperties.from_1_to_0, "epsilon",
        "Доля случайных действий во время обучения")
    gamma: float = MyProperties.get_propertry(
        MyProperties.from_1_to_0, "gamma",
        "Коэффициент доверия опыту (для \"сглаживания\" Q-таблицы)")
    q_alpha: float = MyProperties.get_propertry(
        MyProperties.from_1_to_0, "q_alpha",
        "Скорость обновления Q-таблицы")

    def __init__(self,
                 architecture: Optional[List[int]] = None,
                 add_bias: Optional[bool] = True,
                 name: Optional[str] = None,
                 auto_check_ai: Optional[bool] = True,
                 save_dir: str = "Saves AIs",
                 **kwargs):

        # Альфа коэффициент (коэффициент скорости обучения)
        self.__alpha: float = 1e-2
        # Чем больше, тем скорость и "качество" обучения больше (до определённого момента)
        self.__batch_size: int = 1
        # Какую долю весов "отключаем" при обучении
        self.__number_disabled_weights: float = 0.0
        # Коэффициенты импульса (для оптимизатора Adam)
        self.__impulse1: float = 0.9
        self.__impulse2: float = 0.999
        # Коэффициенты регуляризации
        self.__l1: float = 0
        self.__l2: float = 0

        self.have_bias: bool = add_bias

        self.weights: List[np.matrix] = []  # Появиться после вызова create_weights
        self.biases: List[np.matrix] = []  # Появиться после вызова create_weights
        self._momentums: List[np.matrix] = []
        self._velocities: List[np.matrix] = []

        # Специально убрал аннотиции типов
        self.kit_act_func: ActivationFunctions = ActivationFunctions()
        self.kit_upd_q_table: FuncsUpdateQTable = FuncsUpdateQTable()

        # Какую функцию активации используем
        self.what_act_func: Callable = self.kit_act_func.tanh
        # Какую функцию активации используем для выходных значений
        self.end_act_func: Callable = self.kit_act_func.tanh

        self._packet_delta_weight: List[np.ndarray] = []
        self._packet_delta_biases: List[np.ndarray] = []
        self._packet_layer_answers: List[np.ndarray] = []

        self.q_table: Dict[str, List[float]] = {}
        self.actions: Tuple[str] = ()
        self.__gamma: float = 0
        self.__epsilon: float = 0
        self.__q_alpha: float = 0.1
        self._func_update_q_table: Callable = self.kit_upd_q_table.standart

        # Будем ли совершить случайные действия во время обучения (для "исследования" мира)
        self.recce_mode: bool = False

        self.name: str = name if name else str(np.random.randint(2 ** 31))
        self.save_dir = save_dir

        # Все аргументы из kwargs размещаем каждый в свою переменную
        for item, value in kwargs.items():
            for var_name in self.__dict__:
                # Это может быть коэффициент, поэтому проверяем на имя без "_AI__"
                if item in var_name:
                    self.__dict__[var_name] = value

        # Сразу создаём архитектуру
        if not (architecture is None):
            self.create_weights(architecture, add_bias, **kwargs)
            self.auto_check_ai = auto_check_ai

    def create_weights(self, architecture: List[int], add_bias: bool = True,
                       min_weight: float = -1, max_weight: float = 1, **kwargs):
        """Создаёт матрицу со всеми весами между всеми элементами
        (Подавать надо список с количеством нейронов на каждом слое (архитектуру нейронки))"""

        self.have_bias = add_bias
        self.architecture = architecture

        # Добавляем все веса между слоями нейронов
        for i in range(len(architecture) - 1):
            size = (architecture[i], architecture[i + 1])

            self.weights.append(
                self.kit_act_func.normalize(
                    np.random.random(size=size),
                    min_weight,
                    max_weight,
                )
            )

            if add_bias:
                self.biases.append(
                    self.kit_act_func.normalize(
                        np.random.random(size=(1, architecture[i + 1])),
                        min_weight,
                        max_weight,
                    )
                )
            else:
                self.biases.append(np.zeros((1, architecture[i + 1])))

        # Инициализируем (нулями) штуки для Adam'а
        self._momentums = [0 for _ in range(len(architecture))]
        self._velocities = [0 for _ in range(len(architecture))]

    def genetic_crossing_with(self, ai):
        """ai = Экземпляр такого-же класса AI, как и ЭТА нейронка\n
        Перемешивает веса между ЭТОЙ нейронкой и нейронкой В АРГУМЕНТЕ (50\50) \n
        P.s. Не обязательно, чтобы архитектуры были одинаковы"""

        # В каждом слое
        for layer1, layer2 in zip(self.weights, ai.weights):
            # Для каждого отдельного веса
            for _ in range(layer1.shape[0] * layer1.shape[1]):
                # С шансом 50%
                if np.random.random() < 0.5:
                    # Производим замену на вес из другой матрицы
                    layer1[
                        np.random.randint(layer1.shape[0]),
                        np.random.randint(layer1.shape[1]),
                    ] = layer2[
                        np.random.randint(layer2.shape[0]),
                        np.random.randint(layer2.shape[1]),
                    ]

    def make_mutations(self, mutation: float = 0.01):
        """Создаёт рандомные веса в нейронке
        (Заменяем долю mutation весов на случайные числа)"""

        for layer in self.weights:  # Для каждого слоя
            for _ in range(layer.shape[0] * layer.shape[1]):  # Для каждого элемента
                if np.random.random() <= mutation:  # С шансом mutation
                    # Производим замену на случайное число
                    layer[
                        np.random.randint(layer.shape[0]),
                        np.random.randint(layer.shape[1]),
                    ] = np.random.random() * 2 - 1  # от -1 до 1

    def predict(self, input_data: List[float], _return_answers: bool = False) \
            -> (np.ndarray, Optional[List[np.ndarray]]):
        """Возвращает результат работы нейронки, из входных данных"""
        # Определяем входные данные как вектор
        result_layer = np.array(input_data)

        if result_layer.shape[0] != self.weights[0].shape[0]:
            raise ImpossibleContinue(
                "Размерность входных данных не совпадает с количеством входных нейронов у ИИшки")

        # Сохраняем список всех ответов от нейронов каждого слоя
        list_answers = []

        # Проходимся по каждому слою весов
        layer_count = 0
        for i in range(len(self.weights)):
            layer_count += 1

            if _return_answers:
                ans = np.append(result_layer, 1) \
                    if self.have_bias else result_layer

                list_answers.append(ans)

            # Перемножеаем результат прошлого слоя на слой весов
            result_layer = result_layer.dot(self.weights[i]) + self.biases[i]

            # Процежеваем через функцию активации результат
            # перемножения результата прошлого слоя на слой весов
            if layer_count != len(self.weights):
                result_layer = self.what_act_func(result_layer)

            # Если мы на последнем слое, то пропускаем через конечную функцию активации
            else:
                result_layer = self.end_act_func(result_layer)

        # Если надо, возвращаем спосок с ответами от каждого слоя
        if _return_answers:
            return result_layer, list_answers

        return result_layer

    def learning(self, input_data: List[float], answer: List[float],
                 squared_error: bool = False, use_Adam: bool = True):
        """Метод обратного распространения ошибки для изменения весов в нейронной сети
         (Теперь с оптимизатором Adam)\n"""

        def sign(x: np.ndarray) -> np.ndarray:
            """Возвращает матрицу с +1 или -1 на месте положительного
             или отрицательного числа соответственно"""
            return -1 * (x < 0) + 1 * (x >= 0)

        # Определяем наш ответ как вектор
        answer = np.array(answer)

        # ai_answer | То, что выдала нам нейросеть
        # answers | Список с ответами от каждого слоя нейронов
        ai_answer, answers = self.predict(input_data, True)

        # На сколько должны суммарно изменить веса
        delta_weight: np.ndarray = ai_answer - answer
        delta_bias: np.array = delta_weight
        if squared_error:  # Возводим в квадрат с сохранением знака
            delta_weight = np.power(delta_weight, 2) * sign(delta_weight)

        # Реализуем batch_size
        if len(self._packet_delta_weight) != self.__batch_size:
            # Добавляем ошибки (дельту) с выхода и ответы от слоёв
            self._packet_delta_weight.append(delta_weight)
            self._packet_delta_biases.append(delta_bias)
            self._packet_layer_answers.append(answers)
            return

        # Когда набрали нужное количество складываем все данные
        delta_weight = np.sum(self._packet_delta_weight, axis=0)
        delta_bias = np.sum(self._packet_delta_biases, axis=0)

        # Отдельно складываем ответы от слоёв (т.к. это список векторов)
        answers = [np.array(ans) for ans in self._packet_layer_answers[0]]
        for layer_index in range(len(self._packet_layer_answers[0])):  # Складываем слои отдельно
            for list_answers in self._packet_layer_answers[1:]:  # Первые ответы уже есть
                answers[layer_index] += np.array(list_answers[layer_index])

        self._packet_delta_weight.clear()
        self._packet_delta_biases.clear()
        self._packet_layer_answers.clear()

        # Совершаем всю магию здесь
        for i in range(len(self.weights) - 1, -1, -1):
            # Превращаем векторы в матрицу
            delta_weight = np.matrix(delta_weight)
            delta_bias = np.matrix(delta_bias)
            layer_answer = np.matrix(answers[i])
            weight = self.weights[i]
            bias = self.biases[i]

            # Градиентный спуск
            gradient = delta_weight.T.dot(layer_answer).T

            # L1 и L2 регуляризация
            if self.__l1 or self.__l2:
                grad_sign = sign(np.append(weight, bias, axis=0)) \
                    if self.have_bias else sign(weight)
                gradient += self.__alpha * self.__l1 * grad_sign
                weight *= 1 - self.__alpha * self.__l2
                bias *= 1 - self.__alpha * self.__l2

            # Матрица, предотвращающая переобучение
            # Умножаем изменение веса рандомных нейронов на 0
            # (Отключаем изменение некоторых связей)
            if self.__number_disabled_weights > 0:
                dropout_mask = np.random.random(size=gradient.shape) \
                               >= self.__number_disabled_weights

                gradient = np.multiply(gradient, dropout_mask)

            if use_Adam:
                # Оптимизатор Adam
                self._momentums[i] = self.__impulse1 * self._momentums[i] + \
                                     (1 - self.__impulse1) * gradient
                self._velocities[i] = self.__impulse2 * self._velocities[i] + \
                                      (1 - self.__impulse2) * np.power(gradient, 2)

                momentum = self._momentums[i] / (1 - self.__impulse1)
                velocity = self._velocities[i] / (1 - self.__impulse2)

                # Изменяем веса (С Адамом)
                self.weights[i] -= self.__alpha * momentum[:-1] / np.sqrt(velocity[:-1] + 1e-6)
                self.biases[i] -= self.__alpha * momentum[-1] / np.sqrt(velocity[-1] + 1e-6) \
                    if self.have_bias else 0
            else:
                # Изменяем веса (обычный градиентный спуск)
                # P.s. домножение на веса изменяет их процентно
                gradient = gradient[:-1] if self.have_bias else gradient
                self.weights[i] -= self.__alpha * np.multiply(gradient, np.abs(weight))
                self.biases[i] -= self.__alpha * gradient[-1] if self.have_bias else 0

            if self.have_bias:
                layer_answer = layer_answer[:, :-1]
                delta_bias = np.multiply(self.what_act_func(layer_answer, True),
                                         delta_bias.dot(bias.T))

            # "Переносим" градиент на другой слой (+ умножаем на производную)
            delta_weight = np.multiply(self.what_act_func(layer_answer, True),
                                       delta_weight.dot(weight.T))

    def q_predict(self, input_data: List[float], _return_index_act: bool = False) -> str:
        """Возвращает action, на основе входных данных"""
        ai_result = self.predict(input_data).tolist()

        # "Разведуем окружающую среду" (берём случайное действие)
        if self.__epsilon != 0.0 and np.random.random() < self.__epsilon:
            if _return_index_act:
                return np.random.randint(len(self.actions))
            return self.actions[np.random.randint(len(self.actions))]

        # Находим действие
        if _return_index_act:
            return np.argmax(ai_result)
        return self.actions[np.argmax(ai_result)]

    def make_all_for_q_learning(self, actions: Tuple[str],
                                func_update_q_table: Callable = None,
                                gamma: float = 0.1, epsilon: float = 0.0, q_alpha: float = 0.1):
        """Создаём всё необходимое для Q-обучения
        Q-таблицу (таблица вознаграждений за действие), коэффициент вознаграждениий за будущие действия gamma,\
        коэффициент случайных действий epsilon, и коэффициент скорости изменения Q-таблицы q_alpha

        \n -------------------------

        func_update_q_table - это функция обновления Q-таблицы (выбирается из kit_upd_q_table)

        \n
        \n standart:   Q(s,a) = Q(s,a) + α[r + γ(max Q(s’,a')) - Q(s,a)] \n
        \n future:     Q(s,a) = Q(s,a) + α[r + γ Q(s’,a') - Q(s,a)] \n
        \n future_sum: Q(s,a) = Q(s,a) + α[r + γ(Expected Q(s’,a')) - Q(s,a)] \n
        \n simple:     Q(s,a) = R + γ Q’(s’,a’) \n
        \n simple_max: Q(s,a) = R + γ Q’(s’, max a) \n
        """

        self.actions: Tuple[str] = actions
        if len(self.actions) != self.weights[-1].shape[1]:
            raise ImpossibleContinue(
                "Количество возможных действий (actions) должно"
                "быть равно количеству выходов у нейросети!")

        self.gamma: float = gamma
        self.epsilon: float = epsilon
        self.q_alpha: float = q_alpha

        # Чтобы не указывать будущее состояние, будем обучаться на 1 шаг назад во времени
        self.last_state = None
        self.last_reward = None

        if func_update_q_table is None:
            self._func_update_q_table: Callable = self.kit_upd_q_table.standart
        else:
            self._func_update_q_table: Callable = func_update_q_table

    def q_learning(self, state: List[float],
                   reward: float,
                   learning_method: float = 1,
                   squared_error: bool = False,
                   use_Adam: bool = True,
                   recce_mode: bool = False,
                   ):
        """
        ИИ используется как предсказатель правильных действий\n

        -------------------------- \n

        recce_mode: Режим "исследования окружающей среды" (постоянно выбирать случайное действие)

        -------------------------- \n

        Методы обучения (значение learning_method определяет) : \n
        1 : В качестве "правильного" ответа выбирается то, которое максимально вознаграждается, и на место действия \
        (которое приводит к лучшему ответу) ставиться максимальное значение функции активации \
        (self.act_func.max), а на остальные места минимум функции активации (self.act_func.min) \n
        P.s. Это неочень хорошо, т.к. игнорируются другие варианты, которые приносят либо столько же,
        либо немного меньше  \
        вознаграждения (а вибирается только один "правильный"). НО ОН ХОРОШО ПОДХОДИТ,
        КОГДА У ВАС В ЗАДАЧЕ ИМЕЕТСЯ ИСКЛЮЧИТЕЛЬНО 1 \
        ПРАВИЛЬНЫЙ ОТВЕТ, А "БОЛЕЕ" И "МЕНЕЕ" ПРАВИЛЬНЫХ БЫТЬ НЕ МОЖЕТ \n
        \n

        2 : Делаем ответы которые больше вознаграждаются, более "правильным" \n
        Дробная часть числа означает, в какую степень будем возводить "стремление у лучшим результатам"
        (что это такое читай в P.s.) \
        (НАПРИМЕР: 2.2 означает, что мы используем  метод обучения 2 и возводим в степень 2 \
        "стремление у лучшим результатам", а 2.345 означает, что степень будет равна 3.45 ) \n
        P.s. Работает так: Сначала переводим значения вознаграждений в промежуток от 0 до 1
        (т.е. где вместо максимума вознаграждения\
        - 1, в место минимума - 0, а остальные вознаграждения между ними (без потерь "расстояний" между числами)) \
        потом прибавляем 0.5 и возводим в степень "стремление у лучшим результатам" (уже искажаем "расстояние" между числами) \
        (чтобы ИИ больше стремился именно к лучшим результатам и совсем немного учитывал остальные \
        (чем степень больше, тем меньше учитываются остальные результаты))
        """

        if self.last_state is None:
            self.last_state = state
            self.last_reward = reward
            return

        # (не забываем что мы на 1 шаг в прошлом)
        state_now = state

        # Добовляем новые состояния в Q-таблицу
        last_state_str = str(self.last_state)
        future_state_str = str(state_now)

        default = [0 for _ in range(len(self.actions))]
        self.q_table.setdefault(last_state_str, default)
        self.q_table.setdefault(future_state_str, default)

        # "Режим исследования мира"
        if recce_mode:
            Epsilon, self.__epsilon = self.__epsilon, 1
            self._update_q_table(state_now, reward)
            self.__epsilon = Epsilon
            return

        # Формируем "правильный" ответ
        if learning_method == 1:
            # Заполняем все возможные ответы как неверные
            # (везде ставим минимальное значение конечной функции активации)
            answer = [self.kit_act_func.minimums[str(self.end_act_func)]
                      for _ in range(len(self.actions))]

            # На месте максимального значения из Q-таблицы ставим
            # максимально возможное значение как "правильный" ответ
            answer[np.argmax(self.q_table[last_state_str])] = \
                self.kit_act_func.maximums[str(self.end_act_func)]

        elif 2 < learning_method < 3:
            # Нам нужны значения от минимума функции активации до максимума функции активации
            answer = self.kit_act_func.normalize(np.array(self.q_table[last_state_str]))

            # Искажаем "расстояние" между числами
            answer = answer + 0.5
            answer = np.power(answer, (learning_method - 2) * 10)

            answer = self.kit_act_func.normalize(answer).tolist()

        else:
            raise f"{learning_method}: Неверный метод обучения. " \
                  f"learning_method == 1, или в отрезке: (2; 3)"

        # Изменяем веса
        self.learning(self.last_state, answer, squared_error=squared_error, use_Adam=use_Adam)

        # Обновляем Q-таблицу
        self._update_q_table(state_now, reward)

    def _update_q_table(self, state_now: List[float], reward_for_state: float):
        # Чтобы не указывать будущее состояние, будем обучаться на 1 шаг назад во времени
        state = self.last_state
        state_str = str(self.last_state)
        future_state = state
        future_state_str = str(state)

        ind_act: int = self.q_predict(self.last_state, True)

        all_kwargs = {
            "q_table": self.q_table,
            "q_predict": self.q_predict,
            "q_alpha": self.__q_alpha,
            "reward": self.last_reward,
            "gamma": self.__gamma,
            "state": state,
            "state_str": state_str,
            "future_state": future_state,
            "future_state_str": future_state_str,
            "ind_act": ind_act,
        }

        self.q_table[state_str][ind_act] = self._func_update_q_table(**all_kwargs)

        # Смещаемся на 1 шаг во времени (вперёд)
        self.last_state = state_now
        self.last_reward = reward_for_state

    def check_ai(self):
        """Проверяем Q-таблицу и веса в нейронке на наличие аномальных значений"""
        if not self.auto_check_ai:
            return

        weights_ok, q_table_ok = True, True

        # Проверяем веса (очень грубо)
        for layer_weight in self.weights:
            if np.any(layer_weight > 1e7):
                weights_ok = False

        # Проверяем Q-таблицу, опять таки очень грубо и тупо
        if self.q_table:
            q_negative_nums, q_positive_nums = 1, 0
            for _, string in self.q_table.items():
                q_negative_nums += sum([num < 0 for num in string])
                q_positive_nums += sum([num >= 0 for num in string])

            if q_positive_nums / q_negative_nums < 1:
                q_table_ok = False

        if not weights_ok:
            self.auto_check_ai = False
            print("Веса ИИ слишком большие, рекомендуем уменьшить alpha и пересоздать ИИ")
        if not q_table_ok:
            self.auto_check_ai = False
            print("В Q-таблице отрицательных чисел больше положительных, "
                  "рекомендуем увеличить вознаграждение за хорошие действия и/или уменьшить "
                  "отрицательное вознаграждение для негативных поступков")

    def save(self, ai_name: Optional[str] = None):
        """Сохраняет всю необходимую информацию о текущей ИИ

        (Если не передать имя, то сохранит ИИшку под именем, заданным при создании,
        если передать имя, то сохранит именно под этим)"""
        name_ai = self.name if ai_name is None else ai_name

        # Записываем данны об ИИшке
        def get_name_func(func, kit):
            names_funcs = [f for f in dir(kit)
                           if callable(getattr(kit, f))
                           and not f.startswith("__")]

            func_str = str(func)

            for name_func in names_funcs:
                if name_func in func_str:
                    return name_func

        # Если нет папки для ансамбля, то создаём её
        if not (self.save_dir in listdir(".")):
            mkdir(self.save_dir)

        # Если такое сохранение под таким же именем уже есть,
        # то немного переименовываем текущее имя (как при создании папки в windows)
        if name_ai in listdir(f"{self.save_dir}"):
            # Если это уже не в первый раз, то увеличиваем цифру
            if "(" in name_ai and name_ai[-1] == ")":
                name_ai = name_ai[:-1].split("(")
                name_ai = name_ai[0] + f"({int(name_ai[1]) + 1})"

        ai_data = {}

        ai_data["weights"] = [i.tolist() for i in self.weights]
        ai_data["biases"] = [i.tolist() for i in self.biases]
        ai_data["q_table"] = self.q_table

        # Если используем ансамбль, то сохраняем не имя, а номер
        ai_data["name"] = name_ai.split("/")[-1]

        ai_data["architecture"] = self.architecture
        ai_data["have_bias"] = self.have_bias
        ai_data["actions"] = self.actions

        ai_data["number_disabled_neurons"] = self.__number_disabled_weights
        ai_data["impulse1"] = self.__impulse1
        ai_data["impulse2"] = self.__impulse2
        ai_data["alpha"] = self.__alpha
        ai_data["batch_size"] = self.__batch_size

        ai_data["what_act_func"] = get_name_func(self.what_act_func, self.kit_act_func)
        ai_data["end_act_func"] = get_name_func(self.end_act_func, self.kit_act_func)
        ai_data["func_update_q_table"] = get_name_func(self._func_update_q_table, self.kit_upd_q_table) \
            if self._func_update_q_table else None

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

    def load(self, ai_name: Optional[str] = None):
        """Загружает все данные сохранённой ИИ

        (Если не передать имя, то загрузит сохранение текущей ИИшки,
        если передать имя, то загрузит чужое сохранение)"""
        name_ai = self.name if ai_name is None else ai_name

        def get_func_with_name(name, kit):
            if name is None:
                return None

            return getattr(kit, name)

        # Записываем данны об ИИшке
        try:
            with open(f"Saves AIs/{name_ai}.json", "r") as save_file:
                ai_data = json.load(save_file)

            self.weights = [np.array(i) for i in ai_data["weights"]]
            self.biases = [np.array(i) for i in ai_data["biases"]]
            self.architecture = ai_data["architecture"]
            self.have_bias = ai_data["have_bias"]

            self.number_disabled_weights = ai_data["number_disabled_neurons"]
            self.impulse1 = ai_data["impulse1"]
            self.impulse2 = ai_data["impulse2"]
            self.alpha = ai_data["alpha"]
            self.batch_size = ai_data["batch_size"]

            self.what_act_func = get_func_with_name(
                ai_data["what_act_func"], self.kit_act_func)
            self.end_act_func = get_func_with_name(
                ai_data["end_act_func"], self.kit_act_func)
            self._func_update_q_table = get_func_with_name(
                ai_data["func_update_q_table"], self.kit_upd_q_table)

            self.q_table = ai_data["q_table"]
            self.actions = ai_data["actions"]
            self.gamma = ai_data["gamma"]
            self.epsilon = ai_data["epsilon"]
            self.q_alpha = ai_data["q_alpha"]

            # Переинициализируем штуки для Adam'а
            for i in range(len(self.architecture) - 1):
                size = (self.architecture[i] + self.have_bias, self.architecture[i + 1])
                self._momentums.append(np.zeros(size))
                self._velocities.append(np.zeros(size))

        except FileNotFoundError:
            print(f"Сохранение {name_ai} не найдено")

    def delete(self, ai_name: Optional[str] = None):
        """Удаляет сохранение

        (Если не передать имя, то удалит сохранение текущей ИИшки,
        если передать имя, то удалит другое сохранение)"""
        name_ai = self.name if ai_name is None else ai_name

        try:
            remove(f"Saves AIs/{name_ai}.json")
        except FileNotFoundError:
            pass

    def update(self, ai_name: Optional[str] = None,
               check_ai: bool = True):
        """Обновляем сохранение и проверяем ИИшку (даём непрошеных советов)

        (Если не передать имя, то обновить сохранение текущей ИИшки,
        если передать имя, то обновить другое сохранение)"""

        self.delete(ai_name)
        self.save(ai_name)

        self.auto_check_ai = check_ai
        if self.auto_check_ai:
            self.check_ai()

    def print_parameters(self):
        """Выводит в консоль в формате:
        Параметров: 123456\t\t [1, 2, 3, 4]\t\t + нейрон смещения"""

        all_parameters = sum([i.shape[1] for i in self.biases])
        for layer in self.weights:
            all_parameters += layer.shape[0] * layer.shape[1]

        print(f"{self.name}\t\t",
              f"Параметров: {all_parameters}\t\t",
              f"{self.architecture}", end="")

        if self.have_bias:
            print(" + нейрон смещения", end="")

        print()
