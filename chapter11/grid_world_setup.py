import numpy as np

np.set_printoptions(2, suppress=True)

tsr = 11
tsp = 7
int_r_st = 6
n_state = 12


def create_grid():
    state_vec = np.arange(n_state)
    # reward
    r = np.zeros(state_vec.shape)
    # r[tsr] = 1
    # r[tsp] = -1
    r[tsr] = 2
    r[tsp] = -2
    r[int_r_st] = 1

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
    return TM, r, ex_vals


if __name__ == "__main__":
    _, _, ex_vals = create_grid()
    print(ex_vals)
