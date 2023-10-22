from .Code_My_AI import AI
from typing import List, Optional, Dict, Tuple, Callable
import numpy as np


class AI_with_ensemble(AI):
    def __init__(self, amount_ais: int, *args, **kwargs):
        """Создаёт множество ИИшек"""
        self._ais = [AI(*args, **kwargs) for _ in range(amount_ais)]

    # Суть в том, что для ансамбля ИИшек надо отдельно переопределять функции,
    # чтобы оно всё просто работало

    def __getattr__(self, item):
        """Если мы хотим взять атрибут, который мы в ЭТОМ классе не переопределили
         (т.е. он не мешает механике ансамбля), значит этот атрибут может
         быть в AI, откуда мы его и достаём"""

        return self.__dict__.setdefault(item, self._ais[0].__dict__[item])

    def start_work(self, input_data: List[float], _return_answers: bool = False) \
            -> list[np.ndarray, Optional[List[np.ndarray]] ]:
        """Тот же start_work, но возвращаем предсказание от каждой ИИшки"""

        all_predicts, all_answers = [], []

        # Добавляем все результаты нейронок в один список
        for ai in self._ais:
            if _return_answers:
                # Добавляем и список с ответами, если мы его хотим получить
                predict, answers = ai.start_work(input_data, True)
                all_predicts.append(predict)
                all_answers.append(answers)
            else:
                predict = ai.start_work(input_data)
                all_predicts.append(predict)

        if _return_answers:
            return all_predicts, all_answers
        return all_predicts

    def learning(self, input_data: List[float], answer: List[float],
                 squared_error: bool = False):
        """Обучаем каждую ИИшку ансамбля"""

        # Тут все просто ¯\_(._.)_/¯
        for ai in self._ais:
            ai.learning(input_data, answer, squared_error)

    def make_all_for_q_learning(self, actions: Tuple[str],
                                func_update_q_table: Callable = None,
                                gamma: float = 0.1, epsilon: float = 0.0, q_alpha: float = 0.1):
        """make_all_for_q_learning для каждой ИИшки"""

        # Тут все просто ¯\_(._.)_/¯
        for ai in self._ais:
            ai.make_all_for_q_learning(actions, func_update_q_table, gamma, epsilon, q_alpha)

    def q_start_work(self, input_data: List[float], _return_index_act: bool = False) -> str:
        """Большинство принимает решение"""

        # Собираем голоса от каждой ИИшки (а не как у нас в стране)
        votes = {}
        for ai in self._ais:
            vote = ai.q_start_work(input_data)
            votes.setdefault(vote, 0)

            # +1 ИИшка проголосовала за какое-то действие
            votes[vote] += 1

        # Смотрим что набрало больше всего голосов
        result = max(votes, key=votes.get)

        # Возвращаем индекс или action
        print(votes)
        if _return_index_act:
            return self.actions.index(result)
        return result


