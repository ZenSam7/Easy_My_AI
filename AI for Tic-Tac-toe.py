from time import time

from Games import TicTacToe
from easymyai import AI

size = 3
tic_tac = TicTacToe(size, 3, False, 100)

# Иницииализируем сразу 2 одинаковые нейронки
ais = []
for _ in range(2):
    ai = AI([size ** 2, 100, 100, size ** 2], name=f"tic-tac-toe-{_}")
    ai.main_act_func = ai.kit_act_func.tanh
    ai.end_act_func = ai.kit_act_func.softmax
    actions = tuple((_, __) for _ in range(size) for __ in range(size))
    ai.make_all_for_q_learning(actions, ai.kit_upd_q_table.standart, 0.3, 0.01)
    ais.append(ai)

ai_tic, ai_tac = ais

# ais[0].load()
# ais[1].load()


iters_per_epoch = 10_000
for epoch in range(1, 10 ** 10):
    count_wins = 0
    start_time = time()

    for _ in range(iters_per_epoch):
        for ai in ais:  # Отдельно для крестиков и отдельно для ноликов
            reward = 0
            field = tic_tac.get_field()
            move = ai.q_predict(field)

            win, busy = tic_tac.make_move(move[0], move[1])

            # Клетка уже занята
            if busy:
                reward = -10

            if win:
                count_wins += 1
                reward = 20

            ai.q_learning(field, reward)

    time_spent = int(time() - start_time)
    mean_wins = round(count_wins / iters_per_epoch, 3)
    amount_states = len(ais[0].q_table)
    print(f"{epoch}\t{time_spent}s\t\t{mean_wins}\t{amount_states}")

    # ai.save()
