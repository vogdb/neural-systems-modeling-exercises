import numpy as np
import matplotlib.pyplot as plt
from grid_world_setup import *

n_its = 1000
TM, r, ex_vals = create_grid()

est_vals = np.zeros(n_state)
est_vals[tsr] = r[tsr]
est_vals[tsp] = r[tsp]
count = np.zeros(n_state)
rms = []

for i in range(n_its):
    is_terminal = False
    state = 0
    trj = [state]
    # build a trajectory
    while not is_terminal:
        state_conn = TM[:, state]
        next_state_list = np.argwhere(state_conn != 0).flatten()
        next_state = np.random.choice(next_state_list)
        state = next_state
        trj.append(state)
        if next_state == tsr or next_state == tsp:
            is_terminal = True
    # calculate means
    # reverse trajectory
    trj = np.array(trj)[::-1]
    trj_len = len(trj)
    # trajectory rewards
    trj_rwrd = np.zeros(trj_len)
    trj_rwrd[trj == tsr] = r[tsr]
    trj_rwrd[trj == tsp] = r[tsp]
    trj_rwrd[trj == int_r_st] = r[int_r_st]
    rtg = 0
    for tr in range(trj_len):
        rtg += trj_rwrd[tr]
        state = trj[tr]
        est_vals[state] = est_vals[state] + (rtg - est_vals[state]) / (count[state] + 1)
        count[state] += 1
    if i % 10 == 0:
        rms.append(np.sqrt(np.mean((est_vals - ex_vals) ** 2)))

print('Exact vals:')
print(ex_vals)
print('Trained vals:')
print(est_vals)

plt.plot(range(0, n_its, 10), rms)
plt.show()
