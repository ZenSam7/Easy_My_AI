import numpy as np


class AI:
    """Набор функций для работы с самодельным ИИ"""
    def __init__(self):
        self.matrix_weights = []      # Появиться после вызова create_weights
        self.architecture = []        # Появиться после вызова create_weights
        self.alpha_factor = 0.1       # Альфа каэффицент (каэффицент скорости обучения)
        self.what_activation_function = self.activation_function.ISRU  # Какую функцию активации используем



    def create_weights(self, architecture,
                       fractionality=100):
        """Создаёт матрицу со всеми весами между всеми элементами
        (Подавать надо список с количеством нейронов на каждом слое (архитектуру нейрпонки))
        ((Все веса - рандомные числа от -1 до 1))"""

        self.architecture = architecture  # Добавляем архитектуру (что бы было)


        def create_weight(inp, outp):
            """Создаём веса между inp и outp"""
            layer_weights = []

            for _ in range(inp):
                layer_weights.append([])  # Этот список заполним весами
                for _ in range(outp):
                    # Добавляем дробь от -1 до 1
                    layer_weights[-1].append( np.random.randint(-fractionality, fractionality) / fractionality)
            return np.array(layer_weights)

        # Добавляем все веса между слоями нейронов
        for i in range(len(architecture) -1):
            self.matrix_weights.append( create_weight(architecture[i], architecture[i +1]) )


    def start_work(self, input_data: list):
        """Возвращает результат работы нейронки, из входных данных"""

        # Определяем входные данные как вектор
        result_layer_neurons = np.array(input_data)

        # Проходимся по каждому слою и ...
        # ... Процежеваем через функцию активации ...
        # ... Результат перемножения весов на результат прошлого слоя
        for layer_weight in self.matrix_weights:
            result_layer_neurons = self.what_activation_function(
                np.dot(result_layer_neurons, layer_weight) )

        return result_layer_neurons


    def learning(self, result, answer):
        pass


    def save_data(self, name_this_ai: str):
        """Сохраняет все переменные текущей ИИ"""

        with open("Data of AIs.txt", "a+") as file:
            file.write("name " + name_this_ai + "\n")

            file.write("matrix_weights " +
                       "".join((str( [i.tolist() for i in self.matrix_weights] ).split()))
                       + "\n")
            file.write("architecture " +
                       "".join((str(self.architecture).split()))
                       + "\n")
            file.write("alpha_factor " + str(self.alpha_factor) + "\n")
            file.write("what_activation_function " + str(self.what_activation_function) + "\n")



            file.write("\n")
        print("Все данные сохранены ✔")


    def find_among_data(self, start_with_name="", what_find=""):
        """Ищет среди сохранённых данных нужное нам слово, и возвращает его значение"""
        """!! ВНИМАНИЕ !! Она возвращает ПЕРВОЕ значение (ПЕРВОГО имени), которое нашла"""

        with open("Data of AIs.txt") as file:
            find_name = False

            for line in file:
                # Если нашли название, то записываем данные

                if not find_name and start_with_name == line[5 : 5 + len(start_with_name) ]:
                    find_name = True

                if find_name and what_find == line[0:len(what_find)]:
                    value = line[len(what_find) + 1:-1]

                    if value[0] == "[":      # Либо список
                        from json import loads
                        return loads(value)
                    elif value[0].isdigit(): # Либо число
                        return int(value)
                    else:                    # Либо название
                        return str(value)




    class activation_function:
        """Набор функций активации"""
        def ReLU(x):
            """ReLU"""
            if x < 0:
                return 0.1 * x
            else:
                return x

        def ReLU_2(x):
            """Таже ReLU, но немного другая"""
            if x < 0:
                return 0.1 * x
            elif x <= 1:    # 0 <= x <= 1
                return x
            else:
                return 0.1 * x +0.9

        def Gaussian(x):
            """Распределение Гаусса"""
            return np.e ** (- x**2)

        def SoftPlus(x):
            """ 'Типа экспонента' """
            return np.log(1 + np.e ** x)

        def Curved(x):
            """Как ReLU, только плавная"""
            return ( np.sqrt(x**2 +1) -1 )/2 +x

        def ISRU(x):
            """Похожа на √x"""
            return x / np.sqrt(x**2 +1)

        def Sigmoid(x):
            """Сигмоид"""
            return 1 / (1 + np.e ** (-x))


