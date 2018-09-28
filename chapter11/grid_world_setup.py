import numpy as np

np.set_printoptions(2, suppress=True)

n_state = 12
state_vec = np.arange(n_state)
# reward
r = np.zeros(state_vec.shape)
r[11] = 1
r[7] = -1

TM = np.array([
    [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
])

# TM / np.sum(TM, axis=0)
tmp = np.sum(TM, axis=0)[np.newaxis, :]
tmp = np.repeat(tmp, n_state, axis=0)
prb_mtrx = np.divide(TM, tmp, where=tmp > 1e-6)
prb_mtrx[prb_mtrx > 1] = 0

ex_vals = np.linalg.inv(prb_mtrx.T - np.eye(n_state)).dot(-r)
print(ex_vals)
