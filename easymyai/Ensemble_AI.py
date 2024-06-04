import numpy as np

from .Code_My_AI import AI
from typing import List, Optional, Dict, Tuple, Callable
from numpy import ndarray
import os
from inspect import getmembers


class AI_ensemble(AI):
    """Множество ИИшек в одной коробке (ансамбль), которые немного
    отличаются, но обучаются одновременно, и от этого повышается точность правильного выбора
     (т.к. шанс что одновременно ошибётся множество ИИшек меньше, чем если будет только одна ИИ)"""

    def __init__(self, amount_ais: int, *args, **kwargs):
        """Создаёт множество ИИшек"""
        super().__init__(**kwargs)
        self.ais: List[AI] = [AI(**kwargs) for _ in range(amount_ais)]

        self.save_dir = self.ais[0].save_dir

        # Заменяем имена у ИИшек
        self.ensemble_name = self.ais[0].name
        for index, ai in enumerate(self.ais):
            ai.name = "#" + str(index)

        # Вместо того чтобы у каждой ИИшки была одинаковая Q-таблица, можно
        # просто использовать одну единую для всех
        for ai in self.ais:
            ai.q_table = self.ais[0].q_table
            ai.actions = self.actions
        self.q_table = self.ais[0].q_table

        # Декорируем все методы кроме переопределённых
        funcs_only_in_AI = (
            set(getmembers(AI)[4][1])  # Что объявляли в AI
            - set(getmembers(AI_ensemble)[4][1])  # Что объявляли в AI_ensemble
            - set(vars(AI)["__annotations__"])  # Переменные
        )

        for item_name in funcs_only_in_AI:
            if item_name[:2] != "__":
                self.__dict__[item_name] = self._for_all_ais(item_name)

    def __getattr__(self, item):
        """Всё чего нет в этом классе, точно есть в AI"""
        self.__dict__[item] = getattr(self.ais[0], item)
        return self.__dict__[item]

    def _for_all_ais(self, func_name: str):
        """Декоратор, который применяет функцию (по названию) из AI ко всем ИИшкам"""

        def wrap(*args, **kwargs):
            nonlocal func_name
            for ai in self.ais:
                getattr(ai, func_name)(*args, **kwargs)

        return wrap

    def predict(
        self,
        input_data: List[float],
        reverse: bool = False,
        _return_answers: bool = False,
    ) -> [np.matrix]:
        """Тот же start_work, но возвращаем предсказание от каждой ИИшки \n
        P.s. Этот метод может быть использован только пользователем, т.е.
        контретно этот метод не вызывается внутри библиотеки

        reverse: Если True, то мы будем идти от выхода к входу, и подавать
        надо данные, соразмерные выходному вектору"""

        all_predicts, all_answers = [], []

        # Добавляем все результаты нейронок в один список
        for ai in self.ais:
            if _return_answers:
                # Добавляем и список с ответами, если мы его хотим получить
                predict, answers = ai.predict(
                    input_data, reverse=reverse, _return_answers=True
                )
                all_predicts.append(predict)
                all_answers.append(answers)
            else:
                predict = ai.predict(input_data, reverse=reverse)
                all_predicts.append(predict)

        if _return_answers:
            return all_predicts, all_answers
        return all_predicts

    def q_predict(
        self, input_data: List[float], _return_index_act: bool = False
    ) -> str:
        """Большинство принимает решение"""

        # Собираем голоса от каждой ИИшки (а не как у нас в стране)
        votes = {}
        for ai in self.ais:
            vote = ai.q_predict(input_data)
            # Добавляем голос в словарь, если его нету
            votes.setdefault(vote, 0)

            # +1 ИИшка проголосовала за какое-то действие
            votes[vote] += 1

        # Смотрим что набрало больше всего голосов
        result = max(votes, key=votes.get)

        # Возвращаем индекс или action
        if _return_index_act:
            return self.actions.index(result)
        return result

    def save(self, ai_name: Optional[str] = None):
        """Сохраняет всю необходимую информацию об ансамбле

        (Если не передать имя, то сохранит ИИшку под именем, заданным при создании,
        если передать имя, то сохранит именно под этим)"""

        ensemble_name = ai_name or self.ensemble_name

        # Если нет папки для ансамбля, то создаём её
        if not (ensemble_name in os.listdir(f"{self.save_dir}")):
            os.mkdir(f"{self.save_dir}/{ensemble_name}")

        def saving():
            for ai in self.ais:
                ai.save_dir = self.save_dir
                ai.save(f"{ensemble_name}/{ai.name}")

        # Сохраняем ансамбль ЛЮБОЙ ценой
        try:
            saving()

        # Если папка уже существует, то ничего не делаем
        except FileExistsError:
            pass

        except BaseException as e:
            saving()
            raise e

    def load(self, ai_name: Optional[str] = None):
        """Загружает все данные сохранённой ИИ

        (Если не передать имя, то загрузит сохранение текущей ИИшки,
        если передать имя, то загрузит чужое сохранение)"""

        ensemble_name = ai_name or self.ensemble_name

        for ai in self.ais:
            ai.save_dir = self.save_dir
            ai.load(f"{ensemble_name}/{ai.name}")

        # Вместо того чтобы у каждой ИИшки была одинаковая Q-таблица, можно
        # просто использовать одну единую для всех
        for ai in self.ais:
            ai.q_table = self.ais[0].q_table
            ai.actions = self.ais[0].actions
        self.q_table = self.ais[0].q_table
        self.actions = self.ais[0]

    def delete(self, ai_name: Optional[str] = None):
        """Удаляет сохранения ансамбля

        (Если не передать имя, то удалит сохранение текущей ИИшки,
        если передать имя, то удалит другое сохранение)"""

        ensemble_name = ai_name or self.ensemble_name

        try:
            for save in os.listdir(f"{self.save_dir}/{ensemble_name}"):
                os.remove(f"{self.save_dir}/{ensemble_name}/{save}")
            os.rmdir(f"{self.save_dir}/{ensemble_name}")
        except FileNotFoundError:
            pass

    def update(self, ai_name: Optional[str] = None, check_ai: bool = True):
        """Обновляем все ИИшки ансамбля

        (Если не передать имя, то обновить сохранение текущей ИИшки,
        если передать имя, то обновить другое сохранение)"""

        self.delete(ai_name)
        self.save(ai_name)

        if check_ai:
            self.ais[0].check_ai()

    def print_parameters(self):
        print(f"Количество ИИшек в ансамбле {self.ensemble_name}: {len(self.ais)}")

        parameters_ai = sum([i.shape[1] for i in self.ais[0].biases])
        for layer in self.ais[0].weights:
            parameters_ai += layer.shape[0] * layer.shape[1]

        print(
            f"У одного ИИ: \t Параметров {parameters_ai}\t"
            f"{self.ais[0].architecture}",
            end=" ",
        )
        if self.ais[0].have_bias:
            print("+ нейрон смещения")

        print("Всего параметров:", parameters_ai * len(self.ais))
