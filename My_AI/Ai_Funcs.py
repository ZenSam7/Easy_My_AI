
class FuncsUpdateQTable:
    """Формулы для обновления Q-таблицы: \n

    \n standart:   Q(s,a) = Q(s,a) + α[r + γ(max Q(s’,a')) - Q(s,a)] \n
    \n future:     Q(s,a) = Q(s,a) + α[r + γ Q(s’,a') - Q(s,a)] \n
    \n future_sum: Q(s,a) = Q(s,a) + α[r + γ(Expected Q(s’,a')) - Q(s,a)] \n
    \n simple:     Q(s,a) = R + γ Q’(s’,a’) \n
    \n simple_max: Q(s,a) = R + γ Q’(s’, max a) \n
    """

    def standart(self, q_table: Dict[str, List[float]], state_str: str, ind_act: int, future_state_str: str,
                 q_alpha: float, reward_for_state: float, gamma: float, **kwargs):
        return q_table[state_str][ind_act] + \
            q_alpha * (reward_for_state + gamma *
                       max(q_table[future_state_str]) - q_table[state_str][ind_act])

    def future(self, q_table: Dict[str, List[float]], state_str: str, ind_act: int, future_state: List[float],
               future_state_str: str, q_alpha: float, reward_for_state: float, gamma: float,
               q_start_work: Callable, **kwargs):
        return q_table[state_str][ind_act] + \
            q_alpha * (reward_for_state + gamma *
                       q_table[future_state_str][q_start_work(future_state, True)] - q_table[state_str][ind_act])

    def future_sum(self, q_table: Dict[str, List[float]], state_str: str, ind_act: int,
                   future_state_str: str, q_alpha: float, reward_for_state: float, gamma: float, **kwargs):
        return q_table[state_str][ind_act] + q_alpha * \
            (reward_for_state + gamma * sum(q_table[future_state_str]) - q_table[state_str][ind_act])

    def simple(self, q_table: Dict[str, List[float]], future_state_str: str, future_state: List[float],
               reward_for_state: float, gamma: float, q_start_work: Callable, **kwargs):
        return reward_for_state + \
            gamma * q_table[future_state_str][q_start_work(future_state, True)]

    def simple_max(self, q_table: Dict[str, List[float]], future_state_str: str,
                   reward_for_state: float, gamma: float, **kwargs):
        return reward_for_state + gamma * max(q_table[future_state_str])


class ActivationFunctions:
    """Набор функций активации и их производных"""

    def normalize(self, x: np.ndarray, min: float = 0, max: float = 1):
        # Нормализуем от 0 до 1
        result = x - np.min(x)
        if np.max(result) != 0:
            result = result / np.max(result)

        # От min до max
        result = result * (max - min) + min
        return result

    def relu(self, x: np.ndarray, return_derivative: bool = False):
        """Не действует ограничение value_range"""

        if return_derivative:
            return (x > 0)

        return (x > 0) * x

    def relu_2(self, x: np.ndarray, return_derivative: bool = False):
        if return_derivative:
            return (x < 0) * 0.01 + \
                np.multiply(0 <= x, x <= 1) + \
                (x > 1) * 0.01

        return (x < 0) * 0.01 * x + \
            np.multiply(0 <= x, x <= 1) * x + \
            (x > 1) * 0.01 * x

    def softmax(self, x: np.ndarray, return_derivative: bool = False):
        return np.exp(x) / np.sum(np.exp(x))

    def tanh(self, x: np.ndarray, return_derivative: bool = False):
        if return_derivative:
            return 1 / (10 * np.power(np.cosh(.1 * x), 2))

        return np.tanh(.1 * x)

    def sigmoid(self, x: np.ndarray, return_derivative: bool = False):
        if return_derivative:
            return np.exp(-.1 * x) / (10 * np.power(1 + np.exp(-.1 * x), 2))

        return 1 / (1 + np.exp(-.1 * x))
