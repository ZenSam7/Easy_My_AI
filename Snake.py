import pygame
from random import randint


class Snake:
    """Набор функций для создания змеи"""

    def __init__(self, window_width, window_height, cell_size, amount_food,
                 game_over_function=None, eat_apple_function=None, display_game=False):
        self.window_width = window_width
        self.window_height = window_height
        self.cell_size = cell_size

        # Голова - последняя
        self.snake_body = [[0,0], [1,0], [2,0]]
        self.food_coords = []
        self.amount_food = amount_food

        self.need_grow = False
        self.generation = 0        # Номер поколения
        self.num_steps = 0      # Количество шагов
        self.score = 0
        self.scores = [0]

        self.game_over_function = game_over_function  # Запускаем функцию перед рестартом в game_over
        self.eat_apple_function = eat_apple_function

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


    def draw(self, snake_color=(120,130,140), food_color=(160,50,70), background_color=(40,50,60)):
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
        text_SCORE = font.render(f"Поколение #{self.generation}", True, (200,200,200))
        self.wind.blit(text_SCORE, (0, 0))


        pygame.display.update()


    def game_over(self):
        """Сбрасываем все переменные"""
        if self.game_over_function != None:
            self.game_over_function()  # Запускаем функцию, если она есть

        self.scores.append(self.score)

        self.snake_body = [[0,0], [1,0], [2,0]]
        self.food_coords = []
        self.need_grow = False
        self.spawn_food(self.amount_food)
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
        if self.num_steps > 500:  # Если змея сделала слишком много шагов, то убиваем
            self.game_over()


    def get_blocks(self, visibility_range=3):
        """Возвращаем visibility_range ^2 значений, описывающие состояние клетки вокруг головы змеи
            (если еда то 10, если стена то -10, иначе 0)"""
        data = []

        foods = [ [i[0],i[1]] for i in self.food_coords]
        head = [self.snake_body[-1][0], self.snake_body[-1][1]]

        # Записываем все препятствия, от которых можно убиться
        blocks = [ [i[0],i[1]] for i in self.snake_body]
        for i in range(self.window_width // self.cell_size):
            blocks.append([i, -1])                                     # Потолок
            blocks.append([i, self.window_height // self.cell_size])   # Пол
        for i in range(self.window_height // self.cell_size):
            blocks.append([-1, i])                                     # Левая стена
            blocks.append([self.window_width // self.cell_size, i])    # Правая стена


        # Создаём квадрат с областью видимости visibility_range на visibility_range клеток (голова в центре)
        remainder = (visibility_range -1) //2
        for y in range(head[1] -remainder, head[1] +remainder +1):
            for x in range(head[0] - remainder, head[0] + remainder + 1):
                if [x, y] in blocks:
                    data.append(-10)
                elif [x, y] in foods:
                    data.append(10)
                else:
                    data.append(0)


        return data


    def get_score(self):
        Max, Min, Mean = max(self.scores), min(self.scores), sum(self.scores) / len(self.scores)
        return Max, Min, Mean
