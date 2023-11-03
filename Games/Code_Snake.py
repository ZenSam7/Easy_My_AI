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
            max_steps: int = 100,
            display_game: bool = False,
            dead_reward=-100,
            win_reward=100,
    ):
        self.window_width = window_width
        self.window_height = window_height
        self.cell_size = cell_size
        self.amount_walls = amount_walls
        self.dead_reward = dead_reward
        self.win_reward = win_reward

        # Голова - последняя
        self.snake_body = [[0, 0], [1, 0], [2, 0]]
        self.food_coords = []
        self.amount_food = amount_food

        # Добавляем стены
        self.walls_coords = []
        self.respawn_walls()

        self.max_steps = max_steps

        self.need_grow = False
        self.ignore_game_over = False
        self.generation = 0  # Номер поколения
        self.num_steps = 0  # Количество шагов
        self.score = 0
        self.scores = [0]

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

    def collision(self) -> int:
        """Замечаем столкновения с краем экрана, телом, едой"""

        head = self.snake_body[-1]

        # Столкновение с телом
        if head in self.snake_body[:-1]:
            return self.game_over()

        # Столкновение с стенами
        if head in self.walls_coords:
            return self.game_over()

        # Столкновение с едой
        elif head in self.food_coords:
            self.food_coords.remove(head)
            self.need_grow = True
            self.score += 1
            self.num_steps = 0
            self.spawn_food(1)

            return self.win_reward

        # Выход за границу экрана
        if head[0] < 0 or head[0] >= self.window_width // self.cell_size:
            return self.game_over()
        elif head[1] < 0 or head[1] >= self.window_height // self.cell_size:
            return self.game_over()

        # Если змея заполонила весь экран, то мы выиграли
        if len(self.snake_body) >= (self.window_height / self.cell_size) * \
                (self.window_width / self.cell_size) - self.amount_food:
            return self.game_over()

    def spawn_food(self, num_foods: int = None):
        """Создаём еду"""
        num_foods = self.amount_food if num_foods is None else num_foods

        for _ in range(num_foods):
            coords = [
                randint(0, self.window_width // self.cell_size - 1),
                randint(0, self.window_height // self.cell_size - 1),
            ]

            # Если еда заспавнилась в теле или в другой еде или в стене - пересоздаём
            while (coords in self.food_coords) or \
                    (coords in self.snake_body) or \
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
            if (coords not in self.snake_body) and (coords not in self.food_coords) and \
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

    def game_over(self) -> int:
        """Сбрасываем все переменные"""
        # if self.ignore_game_over:
        #     self.ignore_game_over = False
        #     return 0

        self.scores.append(self.score)

        self.need_grow = False
        self.snake_body = [[0, 0], [1, 0], [2, 0]]
        self.food_coords = []
        self.spawn_food(self.amount_food)
        self.respawn_walls()
        self.score = 0
        self.num_steps = 0
        self.generation += 1

        return self.dead_reward

    def step(self, where_want_move: str) -> int:
        """Запускаем одну итерацию змейки"""

        self.move_snake(where_want_move)
        reward = self.collision()

        if self.display_game:
            self.draw()

        # Если змея сделала слишком много шагов, то убиваем
        self.num_steps += 1
        if self.num_steps > self.max_steps:
            return self.game_over()

        return reward if reward else 0

    def get_blocks(self, visibility_range=3):
        """Возвращаем visibility_range ^2 значений, описывающие состояние клетки вокруг головы змеи
        (если еда то 1, если стена то -1, иначе 0)"""

        assert visibility_range % 2 != 0, "visibility_range только нечётное число"
        self.visibility_range = visibility_range

        data = [[0 for _ in range(visibility_range)] for __ in range(visibility_range)]

        head = self.snake_body[-1]
        remainder = (visibility_range - 1) // 2

        # Проходимся по клеткам вокруг головы
        for y in range(head[1] - remainder, head[1] + remainder + 1):
            for x in range(head[0] - remainder, head[0] + remainder + 1):
                point = [x, y]

                if point in self.food_coords:
                    data[y - head[1] + 1][x - head[0] + 1] = 1
                elif any((point in self.snake_body,
                          x < 0, x >= self.window_width // self.cell_size,
                          y < 0, y >= self.window_width // self.cell_size)):
                    data[y - head[1] + 1][x - head[0] + 1] = -1

        # Избавляемся от внутренних списков
        data = sum(data, [])
        return data

    def get_future_state(self, where_want_move):
        snake_body = self.snake_body.copy()
        food_coords = self.food_coords.copy()
        score, generation, num_steps = self.score, self.generation, self.num_steps

        self.move_snake(where_want_move)
        # # Надо, чтобы лишний раз не выщывалась функция game_over_function()
        # self.ignore_game_over = True

        self.collision()
        future_state = self.get_blocks(self.visibility_range)

        self.snake_body, self.food_coords = snake_body, food_coords
        self.score, self.generation, self.num_steps = score, generation, num_steps

        return future_state

    def get_max_mean_score(self):
        try:
            MAX, MEAN = max(self.scores), sum(self.scores) / len(self.scores)
            self.scores.clear()
            return MAX, MEAN

        except ValueError:
            return 0, 0
