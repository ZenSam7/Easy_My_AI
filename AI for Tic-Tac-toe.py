from easymyai import AI, AI_ensemble
from Games import TicTacToe

tic_tac = TicTacToe(5, 3, True, 100)
tic_tac.make_move(0, 0)
tic_tac.make_move(1, 0)
tic_tac.make_move(1, 1)
tic_tac.make_move(1, 1)
while 1:
    tic_tac.draw()
