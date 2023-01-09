
class AI:
    """Набор функций для работы с самодельным ИИ"""
    def __init__(self):
        pass


    def get_weight(self, inputs, layers, neurons_in_layer, outputs,
                   min_weight=-1, max_weight=1, fractionality=1000):
        """Возвращает матрицу со всеми весами между всеми элементами
        (Подавать надо количество каждого элемента)
        ((Все веса - рандомные числа от -1 до 1))"""

        from random import randint

        matrix_weights = []

        def create_weight(inp, outp):
            """Создаём веса между inp и outp"""
            layer_weights = []

            for _ in range(inp):
                layer_weights.append([])  # Заполним весами от входа до нейронов
                for _ in range(outp):
                    layer_weights[-1].append(randint(min_weight * fractionality,  # Добавляем дробь от -1 до 1
                                                     max_weight * fractionality) / fractionality)

            return layer_weights


        # Сначала добавляем веса между входами и первым слоем нейроноыв
        if layers != 0:
            matrix_weights.append(create_weight(inputs, neurons_in_layer))
        else:
            # Если у нас не будет скрытых слоёв ->
            # Возвращаем веса между входом и выходом
            return matrix_weights.append(create_weight(inputs, outputs))


        # Затем добавляем все веса между слоями нейронов
        for _ in range(layers -1):
            matrix_weights.append(create_weight(neurons_in_layer, neurons_in_layer))


        # Затем добавляем все веса между слоем нейронов и выходами
        matrix_weights.append(create_weight(neurons_in_layer, outputs))

        return matrix_weights



    def activation_function_ReLU(self, x):
        """ReLU function"""
        if x < 0:
            return 0.01 * x
        else:
            return x


    def activation_function_ReLU_2(self, x):
        """Типо ReLU function, но немного другая"""
        if x < 0:
            return 0.01 * x
        elif 0 <= x <= 1:
            return x
        else:
            return 0.01 * x



simple_ai = AI()
inputs = [1]
outputs = [0]

matrix_weights = simple_ai.get_weight(len(inputs), 3, 4, len(outputs))
for i in matrix_weights:
    print(i)
    print()
