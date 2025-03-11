import pygame


class TicTacToe:
    def __init__(self, field_size: int = 3, condition_winnings: int = 3,
                 display_game: bool = False, cell_size: int = 120):
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

    def make_move(self, row: int, column: int) -> bool:
        """Делаем ход, row/column = индексы ряда и колонки, возвращает True если игра закончена"""
        # Если пытаемся сделать ход в уже занятую клетку
        if self.field[row][column] != 0:


        self.field[row][column] = self.queue
        # Передаём ход
        self.queue = -self.queue

        # Определяем завершилась ли игра
        for r in range(0, self.field_size - self.condition_winnings):
            for c in range(0, self.field_size - self.condition_winnings):
                # Для ряда
                if sum(self.field[r][c:c + self.condition_winnings]) * \
                        self.queue == -self.condition_winnings:
                    return True
                # Для столбца
                elif sum(self.field[i_r][c]
                         for i_r in range(r, r + self.condition_winnings)) * \
                        self.queue == -self.condition_winnings:
                    return True
                # Диагональ вправо-вниз
                elif sum(self.field[r + i][c + i]
                         for i in range(self.condition_winnings)) * \
                        self.queue == -self.condition_winnings:
                    return True
                # Диагональ влево-вниз
                elif sum(self.field[r + self.condition_winnings - i][c + i]
                         for i in range(self.condition_winnings)) * \
                        self.queue == -self.condition_winnings:
                    return True

        return False

    def __make_window(self):
        """Просто создаём окно"""

        pygame.init()

        self.wind = pygame.display.set_mode((self.width, self.width))
        pygame.display.set_caption("AI for Tic-Tac-Toe")

    def draw_in_console(self):
        symbols = {0: ' ', 1: 'X', -1: 'O'}
        print("  " + " ".join(str(i) for i in range(self.field_size)))
        for i, row in enumerate(self.field):
            print(str(i) + " " + "|".join(symbols[cell] for cell in row))
            if i < self.field_size - 1:
                print("  " + "-" * (self.field_size * 2 - 1))

    def draw(self):
        self.wind.fill((175, 180, 185))
        bias = 10
        thick = 5

        for i in range(1, self.field_size):
            pygame.draw.line(self.wind, (0, 0, 0), (i * self.cell_size, 0), (i * self.cell_size, self.width), thick)
            pygame.draw.line(self.wind, (0, 0, 0), (0, i * self.cell_size), (self.width, i * self.cell_size), thick)

        for row in range(self.field_size):
            for col in range(self.field_size):
                if self.field[row][col] == 1:
                    pygame.draw.line(self.wind, (10, 10, 235),
                                     (col * self.cell_size + bias, row * self.cell_size + bias),
                                     ((col + 1) * self.cell_size - bias, (row + 1) * self.cell_size - bias), thick+2)
                    pygame.draw.line(self.wind, (10, 10, 235),
                                     (col * self.cell_size + bias, (row + 1) * self.cell_size - bias),
                                     ((col + 1) * self.cell_size - bias, row * self.cell_size + bias), thick+2)
                elif self.field[row][col] == -1:
                    pygame.draw.circle(self.wind, (235, 10, 10),
                                       (col * self.cell_size + self.cell_size // 2, row * self.cell_size + self.cell_size // 2),
                                       self.cell_size // 2 - bias, thick+2)

        pygame.display.flip()