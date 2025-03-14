import time

import pygame


class TicTacToe:
    def __init__(
            self,
            field_size: int = 3,
            condition_winnings: int = 3,
            display_game: bool = False,
            cell_size: int = 120,
    ):
        """condition_winnings = сколько значений надо набрать по столбцу/строке/диагонали чтобы выиграть"""
        self.queue = 1  # Чья очередь ходить (-1 = нолики, 1 = крестики)
        self.field_size = field_size
        self.condition_winnings = condition_winnings
        self.display_game = display_game
        self.cell_size = cell_size
        self.width = field_size * cell_size

        # 0 = Нолики, 1 = крестики
        self.field = [[0 for _ in range(field_size)] for __ in range(field_size)]

        if self.display_game:
            self.__make_window()

    def get_field(self):
        return sum(self.field, [])

    def revert_player(self):
        """Меняем все крестики и нолики местами"""
        self.field = [[-i for i in row] for row in self.field]

    def reset(self):
        """Сбрасываем игру"""
        if self.display_game:
            # Если выиграли, то закрашиваем фон в цвет победителя
            who_win = self.queue if self._return_winnings() else 0
            self.draw(_who_is_win=who_win)
            time.sleep(2)

        self.__init__(self.field_size, self.condition_winnings, self.display_game, self.cell_size)

    def make_move(self, row: int, column: int) -> (bool, bool):
        """Делаем ход, row/column = индексы ряда и колонки \n
        первый возвращаемый параметр: True если игра закончена
        второй возвращаемый параметр: True если клетка уже занята (если занята, то ничего не делаем)"""
        # Если пытаемся сделать ход в уже занятую клетку
        if self.field[row][column] != 0:
            return False, True

        self.field[row][column] = self.queue

        if self.display_game:
            self.draw()
            time.sleep(1)

        # Если выиграли
        if self._return_winnings():
            self.reset()
            return True, False

        # Если всё поле занято
        for row in self.field:
            if row.count(0) != 0:
                break
        else:
            self.reset()
            return False, False

        # Передаём ход только если никто не победил
        self.queue = -self.queue
        return False, False

    def _return_winnings(self) -> bool:
        # Для горизонтальных линий
        for r in range(self.field_size):
            for c in range(self.field_size - self.condition_winnings + 1):
                if (
                        sum(self.field[r][c: c + self.condition_winnings])
                        == self.queue * self.condition_winnings
                ):
                    return True

        # Для вертикальных линий
        for r in range(self.field_size - self.condition_winnings + 1):
            for c in range(self.field_size):
                if (
                        sum(
                            self.field[r_i][c]
                            for r_i in range(r, r + self.condition_winnings)
                        )
                        == self.queue * self.condition_winnings
                ):
                    return True

        # Для диагоналей
        for r in range(self.field_size - self.condition_winnings + 1):
            for c in range(self.field_size - self.condition_winnings + 1):
                if (
                        sum(
                            self.field[r + i][c + i] for i in range(self.condition_winnings)
                        )
                        == self.queue * self.condition_winnings
                ):
                    return True
                if (
                        sum(
                            self.field[r + self.condition_winnings - i - 1][c + i]
                            for i in range(self.condition_winnings)
                        )
                        == self.queue * self.condition_winnings
                ):
                    return True

        return False

    def __make_window(self):
        """Просто создаём окно"""

        pygame.init()

        self.wind = pygame.display.set_mode((self.width, self.width))
        pygame.display.set_caption("AI for Tic-Tac-Toe")
        if self.display_game:
            self.draw()

    def draw_in_console(self):
        symbols = {0: " ", 1: "X", -1: "O"}
        print("  " + " ".join(str(i) for i in range(self.field_size)))
        for i, row in enumerate(self.field):
            print(str(i) + " " + "|".join(symbols[cell] for cell in row))

    def draw(self, _who_is_win: int = 0):
        """Красим фон в зависимости от того, кто сейчас выиграл"""
        background = (175, 180, 215) if _who_is_win == 1 else (205, 180, 185) if _who_is_win == -1 else (175, 180, 185)
        self.wind.fill(background)
        bias = 10
        thick = 5

        for i in range(1, self.field_size):
            pygame.draw.line(
                self.wind,
                (0, 0, 0),
                (i * self.cell_size, 0),
                (i * self.cell_size, self.width),
                thick,
            )
            pygame.draw.line(
                self.wind,
                (0, 0, 0),
                (0, i * self.cell_size),
                (self.width, i * self.cell_size),
                thick,
            )

        for row in range(self.field_size):
            for col in range(self.field_size):
                if self.field[row][col] == 1:
                    pygame.draw.line(
                        self.wind,
                        (10, 10, 235),
                        (col * self.cell_size + bias, row * self.cell_size + bias),
                        (
                            (col + 1) * self.cell_size - bias,
                            (row + 1) * self.cell_size - bias,
                        ),
                        thick + 2,
                    )
                    pygame.draw.line(
                        self.wind,
                        (10, 10, 235),
                        (
                            col * self.cell_size + bias,
                            (row + 1) * self.cell_size - bias,
                        ),
                        (
                            (col + 1) * self.cell_size - bias,
                            row * self.cell_size + bias,
                        ),
                        thick + 2,
                    )
                elif self.field[row][col] == -1:
                    pygame.draw.circle(
                        self.wind,
                        (235, 10, 10),
                        (
                            col * self.cell_size + self.cell_size // 2,
                            row * self.cell_size + self.cell_size // 2,
                        ),
                        self.cell_size // 2 - bias,
                        thick + 2,
                    )

        pygame.display.flip()
