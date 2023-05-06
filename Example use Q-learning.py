# import Code_My_AI
# import pygame
# from random import randint
#
#
# class Game:
#     """Функции для простого примера Q-обучения"""
#
#     def __init__(self):
#         self.arena_size = 10  # Размер стороны квадратной арены
#         self.cell_size = 100
#
#         self.agent_coords = [0, 0]
#         self.finish_coords = [self.arena_size -1, self.arena_size -1] # Финиш - права снизу
#         self.footprints = []  # Добавляем все клетки, на которых был агент
#
#         self.walls_coords = []
#         for _ in range(10): # Всего 10 стен
#             self.walls_coords.append([randint(0, self.arena_size -1),
#                                       randint(0, self.arena_size -1)])
#
#
#         self.game_over_function = None
#         self.win_function = None
#
#         self.make_window()
#
#
#     def make_window(self):
#         """Просто создаём окно"""
#
#         pygame.init()
#         pygame.display.quit()  # Что бы лишнего не создавалось
#
#         self.wind = pygame.display.set_mode((self.arena_size *self.cell_size,
#                                              self.arena_size *self.cell_size))
#         pygame.display.set_caption('Example Q-learning AI')
#
#
#     def draw(self, iteration=0):
#         # Фон
#         self.wind.fill((20,30,40))
#
#         # Следы
#         for footprint in self.footprints:
#             pygame.draw.rect(self.wind, (70,100,90),
#                              (footprint[0] * self.cell_size, footprint[1] * self.cell_size,
#                               self.cell_size, self.cell_size))
#
#         # Агент
#         pygame.draw.rect(self.wind, (130, 170, 140),
#                          (self.agent_coords[0] * self.cell_size, self.agent_coords[1] * self.cell_size,
#                           self.cell_size, self.cell_size))
#
#         # Финиш
#         pygame.draw.rect(self.wind, (120, 160, 190),
#                          (self.finish_coords[0] * self.cell_size, self.finish_coords[1] * self.cell_size,
#                           self.cell_size, self.cell_size))
#
#
#         # Стены
#         for wall in self.walls_coords:
#             pygame.draw.rect(self.wind, (50,90,70),
#                              (wall[0] * self.cell_size, wall[1] * self.cell_size,
#                               self.cell_size, self.cell_size))
#
#
#         # Выводим номер поколения
#         font = pygame.font.Font(None, 40)  # Какой шрифт и размер надписи
#         text_SCORE = font.render(f"Поколение #{iteration}", True, (200,200,200))
#         self.wind.blit(text_SCORE, (0, 0))
#
#
#         pygame.display.update()
#
#
#     def collision(self):
#         coords = self.agent_coords
#
#         if coords[0] < 0 or coords[0] >= self.arena_size or\
#            coords[1] < 0 or coords[1] >= self.arena_size:
#             self.game_over()
#
#         elif coords in self.walls_coords:
#             self.game_over()
#
#         elif coords == self.finish_coords:
#             self.win()
#
#
#     def game_over(self):
#         if self.game_over_function != None:
#             self.game_over_function()
#
#         self.agent_coords = [0, 0]
#         self.footprints = [[0, 0]]
#
#
#     def win(self):
#         if self.win_function != None:
#             self.win_function()
#
#         self.agent_coords = [0, 0]
#         self.footprints = [[0, 0]]
#
#
#     def moving(self, where_want_move):
#         self.footprints.append(self.agent_coords)
#
#         if where_want_move == "up":
#             self.agent_coords[1] -= 1
#         elif where_want_move == "right":
#             self.agent_coords[0] += 1
#         elif where_want_move == "left":
#             self.agent_coords[0] -= 1
#         elif where_want_move == "down":
#             self.agent_coords[1] += 1
#
#
#     def step(self, where_want_move: str, iteration):
#         self.moving(where_want_move)
#         self.collision()
#         self.draw(iteration)
#
#
#
# # Создаём ИИ
# ai = Code_My_AI.AI()
# ai.create_weights([6, 10, 4], add_bias_neuron=True)
#
# ai.what_activation_function = ai.activation_function.ReLU_2
# ai.activation_function.value_range(0, 1)
# ai.end_activation_function = ai.activation_function.Tanh
#
# ai.packet_size = 1
#
# ai.alpha = 1e-7
#
#
#
#
# game = Game()
#
# # Состояния - нахождение в какой-либо клетке поля (координаты каждой клетки)
# states = [[i, j] for j in range(-1, game.arena_size +1) for i in range(-1, game.arena_size +1)]
# actions = ["left", "right", "up", "down"]
#
#
# reward = -0.1
# def died():
#     from time import sleep
#     sleep(0.3)
#     reward = -10
# def win():
#     from time import sleep
#     sleep(0.3)
#     reward = 100
#
# game.game_over_function = died
# game.win_function = win
#
#
# ai.make_all_for_q_learning(states, actions,  0.8)  # Гамма = 0.8
#
#
# data = [0 for _ in range(6)]
#
# learn_iteration = 0
# while 1:
#     learn_iteration += 1
#
#
# ###################### ОТВЕТ ОТ НЕЙРОНКИ
#
#     ai_answer = ai.start_work(data).tolist()
#
#     where_move = ""
#     if max(ai_answer) == ai_answer[0]:
#         where_move = "left"
#     elif max(ai_answer) == ai_answer[1]:
#         where_move = "right"
#     elif max(ai_answer) == ai_answer[2]:
#         where_move = "up"
#     elif max(ai_answer) == ai_answer[3]:
#         where_move = "down"
#
#     game.step(where_move, learn_iteration)
#
# ###################### ОБУЧАЕМ   и    ЗАПИСЫВАЕТ ДАННЫЕ В ОТВЕТ
#
#     this_agent_coords = game.agent_coords
#
#     game.step(where_move, learn_iteration)
#     ai.q_learning(this_agent_coords, where_move, reward, game.agent_coords)
#
#     # В данных: X-Y агента; список вознаграждений за действие
#     data = this_agent_coords + ai.q[ai.states.index(game.agent_coords)]
#
#     game.agent_coords = this_agent_coords
#
#
#     # Если не умерли и не победили, то -0.1 (т.е. штрафуем за лишние шаги)
#     # (P.s. reward изменяется в game.win* или game.game_over* (в game.step), и если они не сработали, то reward как был, так и остаётся -0.1)
#     # *точнее в функциях, которые мы им передаём (win и died)
#     reward = -0.1
#
