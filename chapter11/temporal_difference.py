import numpy as np
from grid_world_setup import *

n_its = 1000
lr = 0.1
lr_dcr = 0.999
TM, r, ex_vals = create_grid()

est_vals = np.zeros(n_state)
est_vals[tsr] = r[tsr]
est_vals[tsp] = r[tsp]
count = np.zeros(n_state)

for i in range(n_its):
    is_terminal = False
    state = 0
    trj = [state]
    # build a trajectory
    while not is_terminal:
        state_conn = TM[:, state]
        next_state_list = np.argwhere(state_conn != 0).flatten()
        next_state = np.random.choice(next_state_list)
        count[state] += 1
        est_vals[state] += lr * (lr_dcr ** count[state]) * (r[state] + est_vals[next_state] - est_vals[state])
        state = next_state
        if next_state == tsr or next_state == tsp:
            is_terminal = True

print('Exact vals:')
print(ex_vals)
print('Trained vals:')
print(est_vals)
