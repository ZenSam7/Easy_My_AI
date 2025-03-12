from easymyai import AI, AI_ensemble
from Games import TicTacToe
from random import randint

tic_tac = TicTacToe(5, 2, True, 100)

while 1:
    try:
        if tic_tac.make_move(randint(0, 4), randint(0, 4)):
            print("!")
    except Exception as e:
        print(e)
