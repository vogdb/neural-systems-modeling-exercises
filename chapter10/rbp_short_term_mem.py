from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

n_units = 8
n_in = 2
n_b = 1
b = 1
b_w = -2.5
lr = .1
n_its = 100000
gate_prb = .5
# connection matrix
M = 2 * (np.random.rand(n_units, n_units + n_in + n_b) - .5)
# bias weights
M[:, 0] = b_w
H = np.zeros(M.shape + (n_units,))
msk = np.ones(M.shape)
# mask bias weights
msk[:, 0] = 0


def train(M, H):
    item = .01
    item_pre = item
    gate = 0
    gate_pre = gate
    dout = .01
    dout_pre = dout
    y = item_pre * np.ones(n_units)
    z = np.concatenate(([b, item, gate], y))
    for c in range(1, n_its):
        q = M.dot(z)
        y = 1 / (1 + np.exp(-q))
        # the last unit is the output
        e = dout - y[-1]
        H_pre = H
        H = np.zeros(H.shape)
        for k in range(n_units):
            for l in range(n_units):
                hold = M[k, l + n_in + n_b] * H_pre[:, :, l]
                hold[k, :] = hold[k, :] + z
                H[:, :, k] = H[:, :, k] + hold
            d_squash = y[k] * (1 - y[k])
            H[:, :, k] = d_squash * H[:, :, k]
        deltaM = e * H[:, :, -1]
        deltaM = lr * deltaM * msk
        M = M + deltaM
        item = np.random.rand()
        if gate_pre == 1:
            dout = item_pre
            gate = 0
        else:
            dout = dout_pre
            gate = np.random.rand() < gate_prb
        z = np.concatenate(([b, item, gate], y))
        item_pre = item
        dout_pre = dout
        gate_pre = gate
    return M


def test(M):
    bg = .01
    test_time = 60
    level_list = np.arange(0.1, 1, .1)
    n_level = len(level_list)
    out = np.zeros((n_units, test_time, n_level))
    des_out = np.zeros((n_level, test_time))
    gate = np.zeros(test_time)
    # gate in
    gate[int(test_time / 3)] = 1
    # gate out
    gate[int(2 * test_time / 3)] = 1

    # find responses of each level on the time range of `test_time`
    for l in range(n_level):
        level = level_list[l]
        input = bg * np.ones(test_time)
        input[int(test_time / 3)] = level
        des_out[l, :] = bg
        des_out[l, int(test_time / 3 + 2):int(2 * test_time / 3) + 1] = level
        # initial state of units
        y = bg * np.ones(n_units)
        out[:, 0, l] = y
        z = np.concatenate(([b, input[0], gate[0]], y))
        for t in range(1, test_time):
            q = M.dot(z)
            y = 1 / (1 + np.exp(-q))
            out[:, t, l] = y
            z = np.concatenate(([b, input[t], gate[t]], y))

    t_vec = np.arange(test_time)
    plt.figure(figsize=(12, 8))
    for l in range(n_level):
        axl = plt.subplot(int('33' + str(l + 1)))
        axl.set_ylim(-.2, 1)
        axl.set_xlabel('level {:.1f}'.format(level_list[l]))
        axl.plot(t_vec, des_out[l, :], '--')
        axl.plot(t_vec, out[n_units - 1, :, l], 'r', linewidth=.6)

    plt.show()


try:
    M = np.loadtxt('short_term_mem.txt')
except Exception:
    M = train(M, H)
    np.savetxt('short_term_mem.txt', M)
test(M)
