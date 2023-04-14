import numpy as np

class AI:
    """Набор функций для работы с самодельным ИИ"""
    def __init__(self, ):
        self.matrix_weights = []      # Появиться после вызова create_weights
        self.neurons_in_layers = []   # Появиться после вызова create_weights


    def create_weights(self, architecture,
                       min_weight=-1, max_weight=1, fractionality=100):
        """Создаёт матрицу со всеми весами между всеми элементами
        (Подавать надо список с количеством нейронов на каждом слое (архитектуру нейрпонки))
        ((Все веса - рандомные числа от -1 до 1))"""

        self.neurons_in_layers = architecture  # Добавляем архитектуру (что бы было)


        def create_weight(inp, outp):
            """Создаём веса между inp и outp"""
            layer_weights = []

            for _ in range(inp):
                layer_weights.append([])  # Этот список заполним весами
                for _ in range(outp):
                    layer_weights[-1].append(np.random.randint(min_weight * fractionality,  # Добавляем дробь от -1 до 1
                                                      max_weight * fractionality)  / fractionality)
            return layer_weights

        # Добавляем все веса между слоями нейронов
        for i in range(len(architecture) -1):
            self.matrix_weights.append( create_weight(architecture[i], architecture[i +1]) )


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

        def Classic(x):
            """ y = x """
            return x

        def ISRU(x):
            """Похожа на √x"""
            return x / np.sqrt(x**2 +1)

        def Sigmoid(x):
            """Сигмоид"""
            return 1 / (1 + np.e ** (-x))



simple_ai = AI()

simple_ai.create_weights([2, 3, 3, 1])

for i in simple_ai.matrix_weights:
    print(i)

