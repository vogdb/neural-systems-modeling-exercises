import numpy as np
from grid_world_setup import *

n_its = 100
TM, r, ex_vals = create_grid()
est_vals = np.zeros(n_state)
est_vals[tsr] = r[tsr]
est_vals[tsp] = r[tsp]

for i in range(n_its):
    is_terminal = False
    state = 0
    while not is_terminal:
        state_conn = TM[:, state]
        next_state_list = np.argwhere(state_conn != 0).flatten()
        prob = 1 / np.sum(state_conn)
        est_vals[state] = r[state] + np.sum(prob * est_vals[next_state_list])
        next_state = np.random.choice(next_state_list)
        if next_state == tsr or next_state == tsp:
            is_terminal = True
        else:
            state = next_state

print('Exact vals:')
print(ex_vals)
print('Trained vals:')
print(est_vals)
