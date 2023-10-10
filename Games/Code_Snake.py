import pygame
from random import randint
from time import sleep
from copy import deepcopy

class Snake:
    """Набор функций для создания змеи"""

    def __init__(
        self,
        window_width: int,
        window_height: int,
        amount_food: int,
        amount_walls: int,
        cell_size: int = 100,
        game_over_function: bool = None,
        eat_apple_function: bool = None,
        max_num_steps: int = 100,
        display_game: bool = False,
    ):
        self.window_width = window_width
        self.window_height = window_height
        self.cell_size = cell_size
        self.amount_walls = amount_walls

        # Голова - последняя
        self.snake_body = [[0, 0], [1, 0], [2, 0]]
        self.food_coords = []
        self.amount_food = amount_food

        # Добавляем стены
        self.walls_coords = []
        self.respawn_walls()

        self.max_num_steps = max_num_steps

        self.need_grow = False
        self.ignore_game_over = False
        self.generation = 0  # Номер поколения
        self.num_steps = 0  # Количество шагов
        self.score = 0
        self.scores = [0]

        # Запускаем функцию перед рестартом в game_over
        self.game_over_function = game_over_function
        self.eat_apple_function = eat_apple_function

        # Пока False -> ничего не выводим на экран (просто работаем с цифрами)
        self.display_game = display_game

        self.spawn_food(self.amount_food)  # Создаём еду

        if self.display_game:
            self.make_window()  # Создаём окно

    def make_window(self):
        """Просто создаём окно"""

        pygame.init()
        # pygame.display.quit()  # Что бы лишнего не создавалось

        self.wind = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("Sneak with AI")

    def move_snake(self, where_want_move: str):
        """Двигаем змею, куда хотим двигаться (если это можно)"""

        head = self.snake_body[-1]
        neck = self.snake_body[-2]

        if not self.need_grow:
            self.snake_body.pop(0)

        self.need_grow = False

        if where_want_move == "left" and head[0] - 1 != neck[0]:
            self.snake_body.append([head[0] - 1, head[1]])
        elif where_want_move == "right" and head[0] + 1 != neck[0]:
            self.snake_body.append([head[0] + 1, head[1]])
        elif where_want_move == "up" and head[1] - 1 != neck[1]:
            self.snake_body.append([head[0], head[1] - 1])
        elif where_want_move == "down" and head[1] + 1 != neck[1]:
            self.snake_body.append([head[0], head[1] + 1])
        else:
            # Если хотим двигаться в тело - продолжаем двигаться в
            # противоположную сторону
            self.need_grow = True

            if where_want_move == "left":
                self.move_snake("right")
            elif where_want_move == "right":
                self.move_snake("left")
            elif where_want_move == "up":
                self.move_snake("down")
            elif where_want_move == "down":
                self.move_snake("up")

    def collision(self):
        """Замечаем столкновения с краем экрана, телом, едой"""

        head = self.snake_body[-1]

        # Столкновение с телом
        if head in self.snake_body[:-1]:
            self.game_over()

        # Столкновение с стенами
        if head in self.walls_coords:
            self.game_over()

        # Столкновение с едой
        elif head in self.food_coords:
            self.eat_apple_function()
            self.food_coords.remove(head)
            self.need_grow = True
            self.score += 1
            self.num_steps = 0
            self.spawn_food(1)

        # Выход за границу экрана
        if head[0] < 0 or head[0] >= self.window_width // self.cell_size:
            self.game_over()
        elif head[1] < 0 or head[1] >= self.window_height // self.cell_size:
            self.game_over()

        # Если змея заполонила весь экран, то мы выиграли
        if len(self.snake_body) >= (self.window_height/self.cell_size) *\
                (self.window_width/self.cell_size) -self.amount_food:
            self.eat_apple_function()
            self.game_over()

    def spawn_food(self, num_foods: int = None):
        """Создаём еду"""
        num_foods = self.amount_food if num_foods is None else num_foods

        for _ in range(num_foods):
            coords = [
                randint(0, self.window_width // self.cell_size - 1),
                randint(0, self.window_height // self.cell_size - 1),
            ]

            # Если еда заспавнилась в теле или в другой еде или в стене - пересоздаём
            while (coords in self.food_coords) or\
                  (coords in self.snake_body)  or\
                  (coords in self.walls_coords):
                coords = [
                    randint(0, self.window_width // self.cell_size - 1),
                    randint(0, self.window_height // self.cell_size - 1),
                ]

            self.food_coords.append(coords)

    def respawn_walls(self):
        """Пересоздаём стены"""
        self.walls_coords = []

        while len(self.walls_coords) != self.amount_walls:
            coords = [
                randint(0, self.window_width // self.cell_size - 1),
                randint(0, self.window_height // self.cell_size - 1),
            ]

            # Нльзя чтоы стена оказась в теле змейки или в еде или другой стене
            if (coords not in self.snake_body) and (coords not in self.food_coords) and\
                    (coords not in self.walls_coords):
                self.walls_coords.append(coords)

    def draw(
        self,
        snake_color=(120, 130, 140),
        food_color=(160, 50, 70),
        walls_color=(110, 110, 110),
        background_color=(40, 50, 60),
    ):
        """Рисуем змею и еду, выводим номер поколения"""

        # Фон
        self.wind.fill((background_color))

        # Еда
        for food_cell in self.food_coords:
            pygame.draw.rect(
                self.wind,
                food_color,
                (
                    food_cell[0] * self.cell_size,
                    food_cell[1] * self.cell_size,
                    self.cell_size,
                    self.cell_size,
                ),
            )

        # Стены
        for wall_cell in self.walls_coords:
            pygame.draw.rect(
                self.wind,
                walls_color,
                (
                    wall_cell[0] * self.cell_size,
                    wall_cell[1] * self.cell_size,
                    self.cell_size,
                    self.cell_size,
                ),
            )

        # Змея
        for snake_cell in self.snake_body:
            pygame.draw.rect(
                self.wind,
                snake_color,
                (
                    snake_cell[0] * self.cell_size,
                    snake_cell[1] * self.cell_size,
                    self.cell_size,
                    self.cell_size,
                ),
            )

        # Выводим номер поколения
        font = pygame.font.Font(None, 40)  # Какой шрифт и размер надписи
        text_SCORE = font.render(f"Поколение #{self.generation}", True, (200, 200, 200))
        self.wind.blit(text_SCORE, (0, 0))

        pygame.display.update()

        # Ждём немного, чтобы человек мог понять что происходит
        sleep(0.1)

    def game_over(self):
        """Сбрасываем все переменные"""
        if self.ignore_game_over:
            self.ignore_game_over = False
            return

        if not (self.game_over_function is None):
            self.game_over_function()  # Запускаем функцию, если она есть

        self.scores.append(self.score)

        self.need_grow = False
        self.snake_body = [[0, 0], [1, 0], [2, 0]]
        self.food_coords = []
        self.spawn_food(self.amount_food)
        self.respawn_walls()
        self.score = 0
        self.num_steps = 0
        self.generation += 1

    def step(self, where_want_move: str):
        """Запускаем одну итерацию змейки"""

        self.move_snake(where_want_move)
        self.collision()

        if self.display_game:
            self.draw()

        self.num_steps += 1
        if self.num_steps > self.max_num_steps:  # Если змея сделала слишком много шагов, то убиваем
            self.game_over()

    def get_blocks(self, visibility_range=3):
        """Возвращаем visibility_range ^2 значений, описывающие состояние клетки вокруг головы змеи
        (если еда то 1, если стена то -1, иначе 0)"""

        assert visibility_range % 2 == 1, "visibility_range is not even number " \
                                          "(because  head should be in center)"

        data = []

        foods = deepcopy(self.food_coords)
        head = deepcopy(self.snake_body[-1])

        # Записываем все препятствия, от которых можно убиться
        blocks = deepcopy(self.snake_body) + deepcopy(self.walls_coords)
        for i in range(self.window_width // self.cell_size):
            blocks.append([i, -1])  # Потолок
            blocks.append([i, self.window_height // self.cell_size])  # Пол
        for i in range(self.window_height // self.cell_size):
            # Левая стена
            blocks.append([-1, i])
            # Правая стена
            blocks.append([self.window_width // self.cell_size, i])

        # Создаём квадрат с областью видимости visibility_range на
        # visibility_range клеток (голова в центре)
        remainder = (visibility_range - 1) // 2
        for y in range(head[1] - remainder, head[1] + remainder + 1):
            for x in range(head[0] - remainder, head[0] + remainder + 1):
                if [x, y] in blocks:
                    data.append(-1)
                elif [x, y] in foods:
                    data.append(1)
                else:
                    data.append(0)

        return data

    def get_future_state(self, where_want_move):
        snake_body = self.snake_body.copy()
        food_coords = self.food_coords.copy()
        score, generation, num_steps = self.score, self.generation, self.num_steps

        self.move_snake(where_want_move)
        # Надо, чтобы лишний раз не выщывалась функция game_over_function()
        self.ignore_game_over = True

        self.collision()
        future_state = self.get_blocks()

        self.snake_body, self.food_coords = snake_body, food_coords
        self.score, self.generation, self.num_steps = score, generation, num_steps

        return future_state

    def get_max_mean_score(self):
        MAX, MEAN = max(self.scores), sum(self.scores) / len(self.scores)
        self.scores.clear()
        return MAX, MEAN
