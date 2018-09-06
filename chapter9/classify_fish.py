from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

n_in = 1
n_hid = 40
n_out = 3
lr = .1
b = 1.
n_its = 100000

me = [2, 4, 6]
sd = [1, 1, 1]
pf = [1 / 3, 1 / 3, 1 / 3]
# pf = [6 / 9, 2 / 9, 1 / 9]
cum_prob = np.cumsum(pf)
# print(cum_prob)
V = 2 * np.random.rand(n_hid, n_in + 1) - 1
U = 2 * np.random.rand(n_out, n_hid + 1) - 1

for c in range(n_its):
    # random fish
    fish_idx = np.where(cum_prob >= np.random.rand())[0][0]
    fish_len = np.random.normal() * sd[fish_idx] + me[fish_idx]
    # set fish class
    d = np.zeros(len(pf))
    d[fish_idx] = 1
    x = np.array([fish_len, b])[:, np.newaxis]
    # hidden unit state
    q = V.dot(x)
    y = 1 / (1 + np.exp(-q))
    y = np.vstack((y, 1))
    z = 1 / (1 + np.exp(-U.dot(y)))
    err = d - z.T
    x = x.T
    y = y.T
    z = z.T
    # print(x.shape, y.shape, z.shape)
    zg = (z * (1 - z)) * err
    deltaU = lr * zg.T.dot(y)
    yg = (y * (1 - y)) * (zg.dot(U))
    # print(yg.shape)
    deltaV = lr * yg[:, :n_hid].T.dot(x)
    # print(deltaV.shape)
    U += deltaU
    V += deltaV

max_fish_length = 8
fish_num = 30
test_len_range = np.linspace(0, max_fish_length, fish_num)
test_len = test_len_range[:, np.newaxis]
test_len = np.hstack((test_len, np.ones((test_len.shape[0], 1)) * b))
test_hid = 1 / (1 + np.exp(-V.dot(test_len.T)))
test_hid = test_hid.T
test_hid = np.hstack((test_hid, np.ones((test_hid.shape[0], 1)) * b))
test_out = 1 / (1 + np.exp(-U.dot(test_hid.T)))
test_out = test_out.T

plt.figure(figsize=(14, 8))
ax1 = plt.subplot(111)
ax1.set_title('Bayes vs NN')
ax1.plot(test_len_range, test_out[:, 0], 'ro', label='NN E_x = {}'.format(me[0]))
ax1.plot(test_len_range, test_out[:, 1], 'g1', label='NN E_x = {}'.format(me[1]))
ax1.plot(test_len_range, test_out[:, 2], 'bs', label='NN E_x = {}'.format(me[2]))

# the same by Bayes rule
# compute likelihoods of
plf = np.array([
    (1 / (sd[0] * np.sqrt(2 * np.pi))) * np.exp(-.5 * ((test_len_range - me[0]) / sd[0]) ** 2),
    (1 / (sd[1] * np.sqrt(2 * np.pi))) * np.exp(-.5 * ((test_len_range - me[1]) / sd[1]) ** 2),
    (1 / (sd[2] * np.sqrt(2 * np.pi))) * np.exp(-.5 * ((test_len_range - me[2]) / sd[2]) ** 2),
])
# plf.shape = 3, 30
plf = plf / np.sum(plf, axis=1, keepdims=True)
# evidence for each l
pl = plf[0] * pf[0] + plf[1] * pf[1] + plf[2] * pf[2]

# posterior of fish class given L = l
plf_pst = np.array([
    plf[0] * pf[0] / pl,
    plf[1] * pf[1] / pl,
    plf[2] * pf[2] / pl,
])

ax1.plot(test_len_range, plf_pst[0], 'r.-', label='Bayes E_x = {}'.format(me[0]))
ax1.plot(test_len_range, plf_pst[1], 'g.-', label='Bayes E_x = {}'.format(me[1]))
ax1.plot(test_len_range, plf_pst[2], 'b.-', label='Bayes E_x = {}'.format(me[2]))

ax1.legend(loc='best')

# ax2 = plt.subplot(122)
# ax2.set_title('likelihoods of L given Fish class')
# ax2.plot(test_len_range, plf[0], 'r')
# ax2.plot(test_len_range, plf[1], 'g')
# ax2.plot(test_len_range, plf[2], 'b')

plt.show()
