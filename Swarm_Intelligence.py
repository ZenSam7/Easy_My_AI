from Code_My_AI import AI
from Games import Swarm_Game
import time


# ИИшки создаются у каждого агента отдельтно
world = Swarm_Game.World(1000, 600, 4, 50,
        {"architecture": [4, 15, 15, 1], "add_bias_neuron": True, "alpha": 1e-2, "angle_delta": 0.1})


while 1:
    world.step()
    time.sleep(0.1)
