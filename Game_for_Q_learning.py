import pygame
from random import randint

class Game:
    """Функции для простого примера Q-обучения"""

    def __init__(self, arena_size, num_walls):
        self.arena_size = arena_size  # Размер стороны квадратной арены
        self.cell_size = 100

        self.agent_coords = [0, 0]
        self.finish_coords = [self.arena_size -1, self.arena_size -1] # Финиш - права снизу
        self.footprints = []  # Добавляем все клетки, на которых был агент

        self.walls_coords = []
        for _ in range(num_walls): # Всего 10 стен
            self.walls_coords.append([randint(0, self.arena_size -1),
                                      randint(0, self.arena_size -1)])


        self.game_over_function = None
        self.win_function = None

        self.make_window()


    def make_window(self):
        """Просто создаём окно"""

        pygame.init()
        pygame.display.quit()  # Что бы лишнего не создавалось

        self.wind = pygame.display.set_mode((self.arena_size *self.cell_size,
                                             self.arena_size *self.cell_size))
        pygame.display.set_caption('Example Q-learning AI')


    def draw(self, iteration=0):
        # Фон
        self.wind.fill((20,30,40))

        # Следы
        for footprint in self.footprints:
            pygame.draw.rect(self.wind, (70,100,90),
                             (footprint[0] * self.cell_size, footprint[1] * self.cell_size,
                              self.cell_size, self.cell_size))

        # Агент
        pygame.draw.rect(self.wind, (130, 170, 140),
                         (self.agent_coords[0] * self.cell_size, self.agent_coords[1] * self.cell_size,
                          self.cell_size, self.cell_size))

        # Финиш
        pygame.draw.rect(self.wind, (120, 160, 190),
                         (self.finish_coords[0] * self.cell_size, self.finish_coords[1] * self.cell_size,
                          self.cell_size, self.cell_size))


        # Стены
        for wall in self.walls_coords:
            pygame.draw.rect(self.wind, (50,90,70),
                             (wall[0] * self.cell_size, wall[1] * self.cell_size,
                              self.cell_size, self.cell_size))


        # Выводим номер поколения
        font = pygame.font.Font(None, 40)  # Какой шрифт и размер надписи
        text_SCORE = font.render(f"Поколение #{iteration}", True, (200,200,200))
        self.wind.blit(text_SCORE, (0, 0))


        pygame.display.update()


    def collision(self):
        coords = self.agent_coords

        if coords[0] < 0 or coords[0] >= self.arena_size or\
           coords[1] < 0 or coords[1] >= self.arena_size:
            self.game_over()

        elif coords in self.walls_coords:
            self.game_over()

        elif coords == self.finish_coords:
            self.win()


    def game_over(self):
        if self.game_over_function != None:
            self.game_over_function()

        self.agent_coords = [0, 0]
        self.footprints = [[0, 0]]


    def win(self):
        if self.win_function != None:
            self.win_function()

        self.agent_coords = [0, 0]
        self.footprints = [[0, 0]]


    def moving(self, where_want_move):
        self.footprints.append([i for i in self.agent_coords])

        if where_want_move == "up":
            self.agent_coords[1] -= 1
        elif where_want_move == "right":
            self.agent_coords[0] += 1
        elif where_want_move == "left":
            self.agent_coords[0] -= 1
        elif where_want_move == "down":
            self.agent_coords[1] += 1


    def step(self, where_want_move: str):
        self.moving(where_want_move)
        self.collision()


    def get_future_coords(self, where_want_move):
        future_agent_coords = [i for i in self.agent_coords]

        if where_want_move == "up":
            future_agent_coords[1] -= 1
        elif where_want_move == "right":
            future_agent_coords[0] += 1
        elif where_want_move == "left":
            future_agent_coords[0] -= 1
        elif where_want_move == "down":
            future_agent_coords[1] += 1

        if future_agent_coords[0] < 0 or future_agent_coords[0] >= self.arena_size or \
                future_agent_coords[1] < 0 or future_agent_coords[1] >= self.arena_size:
            future_agent_coords = [0, 0]
        elif future_agent_coords in self.walls_coords:
            future_agent_coords = [0, 0]
        elif future_agent_coords == self.finish_coords:
            future_agent_coords = [0, 0]

        return future_agent_coords