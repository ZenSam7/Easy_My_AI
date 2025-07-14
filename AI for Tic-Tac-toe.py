from time import time

from Games import TicTacToe
from easymyai import AI

DISPLAY_GAME = 1  # 0 || 1
SIZE = 3
CONDITION_WINNING = 3
tic_tac = TicTacToe(SIZE, CONDITION_WINNING, DISPLAY_GAME, 100)

ai = AI(architecture=[SIZE ** 2, 50, 50, SIZE ** 2], name="tic-tac-toe")
ai.main_act_func = ai.kit_act_func.tanh
ai.end_act_func = ai.kit_act_func.softmax

ai.alpha = 1e-3
ai.l1 = 0.0
ai.l2 = 0.0

actions = tuple((_, __) for _ in range(SIZE) for __ in range(SIZE))
ai.make_all_for_q_learning(actions, ai.kit_upd_q_table.standart, 0.3, 0.01)

# Если я захочу посмотреть как обучились нейронки
if DISPLAY_GAME:
    ai.load()

ai.print_parameters()

print("\nЭпоха | Время на эпоху | % выигрышов | Количество состояний в Q-таблице")

iters_per_epoch = 20_000
for epoch in range(1, 10 ** 10):
    count_wins, count_fail = 0, 1
    start_time = time()

    for _ in range(iters_per_epoch):
        field = tic_tac.get_field()
        move = ai.q_predict(field)

        win_fail = tic_tac.make_move(move[0], move[1])
        # Умножаем на знак хода, т.к. если X выиграли, то O проиграли,
        # т.е. если соперник выиграл, то мы проиграли и наоборот
        win_fail *= tic_tac.queue

        if win_fail == 1:
            # Выиграли
            # Учитываем выигрыш только крестиков
            count_wins += 1
            ai.q_learning(field, 100)
            tic_tac.reset()
        elif win_fail == -1:
            # Проиграли (всё поле заполнено после нашего хода)
            count_fail += 1
            ai.q_learning(field, -100)
            tic_tac.reset()
        else:
            # Играем ЗА противника
            tic_tac.revert_player()
            ai.q_learning(field, 0)

    time_spent = int(time() - start_time)
    mean_wins = round(100*count_wins / (count_wins + count_fail), 1)
    amount_states = len(ai.q_table)
    print(f"{epoch}\t{time_spent}s\t\t{mean_wins}\t{amount_states}")

    ai.update()
