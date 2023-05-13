import pygame
from random import randint


class Snake:
    """Набор функций для создания змеи"""

    def __init__(self, window_width, window_height, cell_size, amount_food,
                 game_over_function=None, display_game=False):
        self.window_width = window_width
        self.window_height = window_height
        self.cell_size = cell_size

        # Голова - последняя
        self.snake_body = [[0,0], [1,0], [2,0]]
        self.food_coords = []
        self.amount_food = amount_food

        self.need_grow = False
        self.generation = 0        # Номер поколения
        self.score = 0
        self.alive = True

        self.game_over_function = game_over_function  # Запускаем функцию перед рестартом в game_over

        self.display_game = display_game    # Пока False -> ничего не выводим на экран (просто работаем с цифрами)


        self.spawn_food(self.amount_food)   # Создаём еду

        if self.display_game:
            self.make_window()    # Создаём окно



    def make_window(self):
        """Просто создаём окно"""

        pygame.init()
        #pygame.display.quit()  # Что бы лишнего не создавалось

        self.wind = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption('Sneak with AI')


    def move_snake(self, where_want_move: str):
        """Двигаем змею, куда хотим двигаться (если это можно)"""

        head = self.snake_body[-1]
        neck = self.snake_body[-2]

        if not self.need_grow:
            self.snake_body.pop(0)

        self.need_grow = False


        if where_want_move == "left" and \
            head[0] -1 != neck[0]:
            self.snake_body.append([head[0] -1, head[1]])
        elif where_want_move == "right" and \
            head[0] +1 != neck[0]:
            self.snake_body.append([head[0] +1, head[1]])
        elif where_want_move == "up" and \
            head[1] -1 != neck[1]:
            self.snake_body.append([head[0], head[1] -1])
        elif where_want_move == "down" and \
            head[1] +1 != neck[1]:
            self.snake_body.append([head[0], head[1] +1])
        else:
            # Если хотим двигаться в тело - продолжаем двигаться в противоположную сторону
            self.need_grow = True  # Надо, что бы змея лишний раз не укорачивалась

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

        # Столкновение с едой
        if head in self.food_coords:
            self.food_coords.remove(head)
            self.need_grow = True
            self.score += 1
            self.spawn_food(1)

        # Выход за границу экрана
        if head[0] < 0 or head[0] >= self.window_width // self.cell_size:
            self.game_over()
        if head[1] < 0 or head[1] >= self.window_height // self.cell_size:
            self.game_over()


    def spawn_food(self, num_foods=None):
        """Создаём еду"""
        num_foods = self.amount_food if num_foods == None else num_foods

        for _ in range(num_foods):
            coords = [randint(0, self.window_width // self.cell_size - 1),
                      randint(0, self.window_height // self.cell_size - 1)]

            # Если еда заспавнилась в теле или в другой еде - пересоздаём
            while (coords in self.food_coords) or (coords in self.snake_body):
                coords = [randint(0, self.window_width // self.cell_size - 1),
                          randint(0, self.window_height // self.cell_size - 1)]

            self.food_coords.append(coords)


    def draw(self, iteration, snake_color=(120,130,140), food_color=(160,50,70), background_color=(40,50,60)):
        """Рисуем змею и еду, выводим номер поколения"""

        # Фон
        self.wind.fill((background_color))

        # Еда
        for food_cell in self.food_coords:
            pygame.draw.rect(self.wind, food_color,
                    (food_cell[0] *self.cell_size, food_cell[1] *self.cell_size,
                     self.cell_size, self.cell_size))

        # Змея
        for snake_cell in self.snake_body:
            pygame.draw.rect(self.wind, snake_color,
                             (snake_cell[0] *self.cell_size, snake_cell[1] *self.cell_size,
                              self.cell_size, self.cell_size))

        # Выводим номер поколения
        font = pygame.font.Font(None, 40)  # Какой шрифт и размер надписи
        text_SCORE = font.render(f"Поколение #{iteration}", True, (200,200,200))
        self.wind.blit(text_SCORE, (0, 0))


        pygame.display.update()


    def game_over(self):
        """Сбрасываем все переменные"""
        if self.game_over_function != None:
            self.game_over_function()  # Запускаем функцию, если она есть

        self.snake_body = [[0,0], [1,0], [2,0]]
        self.food_coords = []
        self.need_grow = False
        self.spawn_food(self.amount_food)
        self.score = 0

        self.alive = False


    def step(self, where_want_move: str, iteration):
        """Запускаем одну итерацию змейки"""

        if self.alive:
            self.move_snake(where_want_move)
            self.collision()

        if self.display_game:
            self.draw(iteration)


    def get_range_to_blocks(self) -> list:
        """Записываем минимальное расстояние до стены (или тела змеи) и еды \n
        По 4м осям, относительно головы (влево, вправо, вверх, вниз)"""
        data = []

        range_to_block = 0
        head = [i for i in self.snake_body[-1]]

        head[0] -= 1
        while 0 <= head[0] < self.window_width // self.cell_size and \
                not (head in self.snake_body):
            range_to_block += 1
            head[0] -= 1
        data.append(range_to_block)

        ########## Тоже самое, но для других осей
        range_to_block = 0
        head = [i for i in self.snake_body[-1]]

        head[0] += 1
        while 0 <= head[0] < self.window_width // self.cell_size and \
                not (head in self.snake_body):
            range_to_block += 1
            head[0] += 1
        data.append(range_to_block)

        ########## Тоже самое, но для других осей
        range_to_block = 0
        head = [i for i in self.snake_body[-1]]

        head[1] -= 1
        while 0 <= head[1] < self.window_height // self.cell_size and \
                not (head in self.snake_body):
            range_to_block += 1
            head[1] -= 1
        data.append(range_to_block)

        ########## Тоже самое, но для других осей
        range_to_block = 0
        head = [i for i in self.snake_body[-1]]

        head[1] += 1
        while 0 <= head[1] < self.window_height // self.cell_size and \
                not (head in self.snake_body):
            range_to_block += 1
            head[1] += 1
        data.append(range_to_block)

        ####################

        ####################

        #################### Тоже самое, но для еды

        x_foods = [food[0] for food in self.food_coords]
        y_foods = [food[1] for food in self.food_coords]
        head = [i for i in self.snake_body[-1]]

        if head[1] in y_foods:  # Если голова и еда по значению Y равны, то считаем расстояние

            range_to_eat = 0
            head = [i for i in self.snake_body[-1]]

            while 0 <= head[0] < self.window_width // self.cell_size and \
                    not (head in self.food_coords):
                range_to_eat += 1
                head[0] -= 1
            # Если вылезли за экран, то расстояние = 0
            if 0 <= head[0] < self.window_width // self.cell_size:
                data.append(range_to_eat)
            else:
                data.append(0)


            ########## Тоже самое

            range_to_eat = 0
            head = [i for i in self.snake_body[-1]]

            while 0 <= head[0] < self.window_width // self.cell_size and \
                    not (head in self.food_coords):
                range_to_eat += 1
                head[0] += 1
            # Если вылезли за экран, то расстояние = 0
            if 0 <= head[0] < self.window_width // self.cell_size:
                data.append(range_to_eat)
            else:
                data.append(0)

        else:
            data.append(0)
            data.append(0)



        if head[0] in x_foods:  # Если голова и еда по значению Y равны, то считаем расстояние

            range_to_eat = 0
            head = [i for i in self.snake_body[-1]]

            while 0 <= head[1] < self.window_height // self.cell_size and \
                    not (head in self.food_coords):
                range_to_eat += 1
                head[1] -= 1
            # Если вылезли за экран, то расстояние = 0
            if 0 <= head[1] < self.window_height // self.cell_size:
                data.append(range_to_eat)
            else:
                data.append(0)


            ########## Тоже самое

            range_to_eat = 0
            head = [i for i in self.snake_body[-1]]

            while 0 <= head[1] < self.window_height // self.cell_size and \
                    not (head in self.food_coords):
                range_to_eat += 1
                head[1] += 1
            # Если вылезли за экран, то расстояние = 0
            if 0 <= head[1] < self.window_height // self.cell_size:
                data.append(range_to_eat)
            else:
                data.append(0)

        else:
            data.append(0)
            data.append(0)


        return data
