import pygame
from random import randint
import math
from My_AI.Code_My_AI import AI


class World:
    def __init__(self, window_width, window_height,
                 amount_sources, amount_agents,
                 settings):
        """ai_settings: {architecture, add_bias_neuron, alpha, angle_delta}"""
        # Добавляем Источники
        self.all_sources = [Source(window_width, window_height)
                            for _ in range(amount_sources)]

        # Добавляем Агентов
        self.all_agents = [Agent(window_width, window_height, settings)
                           for _ in range(amount_agents)]

        self.make_window(window_width, window_height)

    def make_window(self, window_width, window_height):
        # Создаём окно
        pygame.init()
        self.window = pygame.display.set_mode((window_width, window_height))
        pygame.display.set_caption("Swarm Intelligence")

    def draw(self):
        """Рисуем Источники и Агентов"""
        # Фон
        self.window.fill((30, 30, 30))

        # Источники
        for source in self.all_sources:
            pygame.draw.circle(
                self.window, source.color, source.coords, source.size
            )

        # Агенты
        for agent in self.all_agents:
            pygame.draw.circle(
                self.window, agent.color, agent.coords, agent.size
            )

        pygame.display.update()

    def step(self):
        """Двигаем Агентов"""
        # Находим координаты всех агентов
        agents_coords = [agent.coords for agent in self.all_agents]
        objects_coords = agents_coords + [source.coords for source in self.all_sources]

        for agent in self.all_agents:
            agent.move(agents_coords)
            agent.learn(objects_coords)

        # Рисуем
        self.draw()


class Source:
    def __init__(self, window_width, window_height):
        self.size = randint(10, 20)
        self.color = (randint(100, 200), randint(100, 200), randint(100, 200))

        self.coords = [randint(0, window_width), randint(0, window_height)]


class Agent:
    def __init__(self, window_width, window_height, settings):
        # angle_delta —— это то, на сколько максимум ИИшка может изменить угол
        # angle_delta ∈ (0; 1]   (от 0° до 180°; от 0 до π)
        # angle_delta = 0 —— ИИшка не может изменять угол
        self.angle_delta = settings["angle_delta"]
        self.size = 2
        self.color = (randint(100, 200), randint(100, 200), randint(100, 200))

        self.borders = [window_width, window_height]
        self.coords = [randint(0, self.borders[0]), randint(0, self.borders[1])]
        self.speed = randint(10, 50)/10
        self.angle_move = randint(-100, 100)/100

        # Создаём ИИшки
        self.ai = AI()
        self.ai.create_weights(settings["architecture"],
                               add_bias_neuron=settings["add_bias_neuron"])
        self.ai.what_act_func = self.ai.kit_act_funcs.tanh
        self.ai.__alpha = settings["alpha"]

    def move(self, agents_coords):
        """Двигаем Агента"""

        # Находим ближайшего соседа
        nearest_neighbor = [0, 0]
        for coords in agents_coords:
            # Сравниваем расстояние до уже найденного ближайшего Агента и
            # до другого Агента, и добавляем соседа с наименьшим расстоянием
            if math.sqrt(
                (self.coords[0] - coords[0]) ** 2 +
                (self.coords[1] - coords[1]) ** 2
            ) < math.sqrt(
                (self.coords[0] - nearest_neighbor[0]) ** 2 +
                (self.coords[1] - nearest_neighbor[1]) ** 2
            ) and coords != self.coords:
                nearest_neighbor = coords

        # Вичисляем изменение поворота от ИИ
        ai_angle = self.ai.predict([*self.coords, *nearest_neighbor])[0] / self.angle_delta
        self.angle_move += ai_angle * math.pi

        # Двигаемся (Если вышли за границу - спавним посередине)
        # По X
        self.coords[0] += self.speed * math.cos(self.angle_move)
        if self.coords[0] < 0:
            self.coords[0] = self.borders[0]/2
        elif self.coords[0] > self.borders[0]:
            self.coords[0] = self.borders[0]/2
        # По Y
        self.coords[1] += self.speed * math.sin(self.angle_move)
        if self.coords[1] < 0:
            self.coords[1] = self.borders[1]/2
        elif self.coords[1] > self.borders[1]:
            self.coords[1] = self.borders[1]/2

    def learn(self, objects_coords):
        """Обучаем каждого агента идти к другим Агентам или к Источникам"""

        # Находим ближайший объект
        nearest_odject = [0, 0]
        for coords in objects_coords:
            if math.sqrt(
                (self.coords[0] - coords[0]) ** 2 +
                (self.coords[1] - coords[1]) ** 2
            ) < math.sqrt(
                (self.coords[0] - nearest_odject[0]) ** 2 +
                (self.coords[1] - nearest_odject[1]) ** 2
            ) and coords != self.coords:
                nearest_odject = coords

        # Угол до ближайшего соседа
        answer_angle = math.atan2(self.coords[1] - nearest_odject[1],
                                  self.coords[0] - nearest_odject[0]) / math.pi
        # Изменение угла
        answer_angle = self.angle_move - answer_angle

        answer_angle /= self.angle_delta

        self.ai.learning([*self.coords, *nearest_odject], answer_angle)
