from Code_My_AI import AI


class AI_with_ensemble(AI):
    def create_ensemble(self, amount_ais: int, *args, **kwargs):
        """Создаёт множество ИИшек"""
        self._ais = [AI(*args, **kwargs) for _ in range(amount_ais)]



def __for_all_ais(self, func):
    """Для каждой ИИшки вызывает func"""
    for ai in self._ais:
        func(ai)
