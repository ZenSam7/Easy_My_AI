from typing import Dict, List, Callable
import numpy as np


class FuncsUpdateQTable:
    """Формулы для обновления Q-таблицы: \n

    \n standart:   Q(s,a) = Q(s,a) + α[r + γ(max Q(s’,a')) - Q(s,a)] \n
    \n future:     Q(s,a) = Q(s,a) + α[r + γ Q(s’,a') - Q(s,a)] \n
    \n future_sum: Q(s,a) = Q(s,a) + α[r + γ(Expected Q(s’,a')) - Q(s,a)] \n
    \n simple:     Q(s,a) = R + γ Q’(s’,a’) \n
    \n simple_max: Q(s,a) = R + γ Q’(s’, max a) \n
    """

    def standart(self, q_table: Dict[str, List[float]], state_str: str,
                 ind_act: int, future_state_str: str, q_alpha: float,
                 reward: float, gamma: float, **kwargs) -> float:
        return q_table[state_str][ind_act] + \
            q_alpha * (reward + gamma *
                       max(q_table[future_state_str]) - q_table[state_str][ind_act])

    def future(self, q_table: Dict[str, List[float]], state_str: str, ind_act: int,
               future_state: List[float], future_state_str: str, q_alpha: float,
               reward: float, gamma: float, q_predict: Callable, **kwargs) -> float:
        return q_table[state_str][ind_act] + \
            q_alpha * (reward + gamma *
                       q_table[future_state_str][q_predict(future_state, True)] -
                       q_table[state_str][ind_act])

    # def future_sum(self, q_table: Dict[str, List[float]], state_str: str,
    #                ind_act: int, future_state_str: str, q_alpha: float,
    #                reward: float, gamma: float, **kwargs) -> float:
    #     return q_table[state_str][ind_act] + q_alpha * \
    #         (reward + gamma * sum(q_table[future_state_str]) - q_table[state_str][ind_act])

    def simple(self, q_table: Dict[str, List[float]], future_state_str: str,
               future_state: List[float], reward: float, gamma: float,
               q_predict: Callable, **kwargs) -> float:
        return reward + \
            gamma * q_table[future_state_str][q_predict(future_state, True)]

    def simple_max(self, q_table: Dict[str, List[float]], future_state_str: str,
                   reward: float, gamma: float, **kwargs) -> float:
        return reward + gamma * max(q_table[future_state_str])


class ActivationFunctions:
    """Набор функций активации и их производных"""
    def __init__(self):
        self.minimums = {
            str(self.tanh): -1,
            str(self.softmax): 0,
            str(self.sigmoid): 0,
            str(self.relu): 0,
            str(self.relu_2): 0,  # ?
            str(None): -100
        }

        self.maximums = {
            str(self.tanh): 1,
            str(self.softmax): 1,
            str(self.sigmoid): 1,
            str(self.relu): 100,  # ?
            str(self.relu_2): 1,  # ?
            str(None): 100,
        }

    def normalize(self, x: np.ndarray, min: float = 0, max: float = 1) -> np.ndarray:
        # Нормализуем от 0 до 1
        result = x - np.min(x)
        if np.max(result) != 0:
            result = result / np.max(result)

        # От min до max
        result = result * (max - min) + min
        return result

    def relu(self, x: np.ndarray, return_derivative: bool = False) -> np.ndarray:
        if return_derivative:
            return (x > 0)

        return (x > 0) * x

    def relu_2(self, x: np.ndarray, return_derivative: bool = False) -> np.ndarray:
        if return_derivative:
            return (x < 0) * 0.01 + \
                np.multiply(0 <= x, x <= 1) + \
                (x > 1) * 0.01

        return (x < 0) * 0.01 * x + \
            np.multiply(0 <= x, x <= 1) * x + \
            (x > 1) * 0.01 * x

    def softmax(self, x: np.ndarray, return_derivative: bool = False) -> np.ndarray:
        return np.exp(x) / np.sum(np.exp(x))

    def tanh(self, x: np.ndarray, return_derivative: bool = False) -> np.ndarray:
        if return_derivative:
            return 1 / np.power(np.cosh(x), 2)

        return np.tanh(x)

    def sigmoid(self, x: np.ndarray, return_derivative: bool = False) -> np.ndarray:
        if return_derivative:
            return np.exp(-x) / np.power(1 + np.exp(-x), 2)

        return 1 / (1 + np.exp(-x))