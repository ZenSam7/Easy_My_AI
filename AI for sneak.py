import Code_My_AI

import pygame
from random import randint


class Snake:
    """Набор функций для создания змеи"""

    def __init__(self, window_width, window_height, cell_size, amount_food):
        self.window_width = window_width
        self.window_height = window_height
        self.cell_size = cell_size

        # Голова - последняя
        self.snake_body = [[0,0], [1,0], [2,0]]
        self.food_coords = []
        self.amount_food = amount_food

        self.need_grow = False


        pygame.init()
        pygame.display.quit()  # Что бы лишнего не создавалось

        """Просто создаём окно"""
        self.wind = pygame.display.set_mode((window_width, window_height))
        pygame.display.set_caption('Sneak with AI')

        self.spawn_food(self.amount_food)


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


    def draw(self, snake_color=(120,130,140), food_color=(160,50,70), background_color=(40,50,60)):
        """Рисуем змею и еду"""

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


        pygame.display.update()


    def game_over(self, func=None):
        """Сбрасываем все переменные"""
        if func != None:
            func()

        self.snake_body = [[0,0], [1,0], [2,0]]
        self.food_coords = []
        self.need_grow = False
        self.spawn_food(self.amount_food)


    def step(self, where_want_move: str):
        """Запускаем одну итерацию змейки"""

        self.move_snake(where_want_move)
        self.collision()
        self.draw()


def get_range_to_blocks(snake) -> list:
    """Записываем минимальное расстояние до стены или тела змеи
    По 4м осям, относительно головы (влево, вправо, вверх, вниз)"""
    data = []



    range_to_block = 0
    head = [i for i in snake.snake_body[-1]]  # Копируем координаты головы

    head[0] -= 1
    while 0 <= head[0] < snake.window_width // snake.cell_size and \
        not (head in snake.snake_body):
        range_to_block += 1
        head[0] -= 1
    data.append(range_to_block)

    ########## Тоже самое, но для других осей
    range_to_block = 0
    head = [i for i in snake.snake_body[-1]]

    head[0] += 1
    while 0 <= head[0] < snake.window_width // snake.cell_size and \
        not (head in snake.snake_body):
        range_to_block += 1
        head[0] += 1
    data.append(range_to_block)

    ########## Тоже самое, но для других осей
    range_to_block = 0
    head = [i for i in snake.snake_body[-1]]

    head[1] -= 1
    while 0 <= head[1] < snake.window_height // snake.cell_size and \
        not (head in snake.snake_body):
        range_to_block += 1
        head[1] -= 1
    data.append(range_to_block)

    ########## Тоже самое, но для других осей
    range_to_block = 0
    head = [i for i in snake.snake_body[-1]]

    head[1] += 1
    while 0 <= head[1] < snake.window_height // snake.cell_size and \
        not (head in snake.snake_body):
        range_to_block += 1
        head[1] += 1
    data.append(range_to_block)


    ####################

    #################### Тоже самое, но для еды

    head = [i for i in snake.snake_body[-1]]  # Копируем координаты головы

    for food_coord in snake.food_coords: # Для каждой еды отдельно
        # Если по оси Y мы на линии с едой, то добавляем расстояние влево и вправо
        if head[1] == food_coord[1]:
            data.append(food_coord[0] - head[0])
        else:
            data.append(0)

        # Если по оси X мы на линии с едой, то добавляем расстояние влево и вправо
        if head[0] == food_coord[0]:
            data.append(food_coord[1] - head[1])
        else:
            data.append(0)


    return data




# Создаём Змейку
snake = Snake(1200, 900, 100, 3)


# Создаём ИИ
ai = Code_My_AI.AI()
ai.create_weights([10, 25, 15, 4], add_bias_neuron=False)

ai.what_activation_function = ai.activation_function.ReLU_2
ai.activation_function.value_range(0, 1)
ai.end_activation_function = ai.activation_function.Tanh

ai.packet_size = 10

ai.alpha = 1e-5



# Загружаем последнее сохранение
# ai.load_data("Snake")

learn_iteration = 0
while 1:
    learn_iteration += 1


######################## ЗАПИСЫВАЕТ ДАННЫЕ В ОТВЕТ

    data = get_range_to_blocks(snake)

######################## ОТВЕТ ОТ НЕЙРОНКИ

    ai_answer = ai.start_work(data).tolist()
    if max(ai_answer) == ai_answer[0]:
        snake.step("left")
    elif max(ai_answer) == ai_answer[1]:
        snake.step("right")
    elif max(ai_answer) == ai_answer[2]:
        snake.step("up")
    elif max(ai_answer) == ai_answer[3]:
        snake.step("down")

######################## ОБУЧАЕМ


    if learn_iteration % 1000 == 0:
        ai.delete_data("Snake")
        ai.save_data("Snake")


