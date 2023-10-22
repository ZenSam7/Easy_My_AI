from My_AI import AI_with_ensemble, AI

ai = AI_with_ensemble(100, [2, 50, 50, 4], True, "TEST")
ai.make_all_for_q_learning(("l", "r", "d", "u"), ai.kit_upd_q_table.standart, 0.1, 0.01, 0.1)

ai.alpha = 1e-5

# AI([2, 5, 5, 3], True, "TEST").make_all_for_q_learning(("l", "r"), ai.kit_upd_q_table.standart, 0.1, 0.01, 0.1)

print(ai.q_start_work([42, 42], True), ai.actions)
