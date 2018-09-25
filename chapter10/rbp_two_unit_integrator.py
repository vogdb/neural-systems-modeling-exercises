from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

n_out = 2
lr = .005
n_its = 100
v_abs = .5  # input-to-hidden absolute weight
w_abs = .495  # hidden-recurrent absolute weight
V = v_abs * np.array([
    [1, -1],
    [-1, 1],
])
W = w_abs * np.array([
    [1, -1],
    [-1, 1],
])
M = np.hstack((V, W))
print('Expected weights:')
print(M)

# generating expected outputs
t_end = 500
x_hold = np.zeros((2, t_end + 1))
x_hold[0, int(t_end / 10 + 1)] = 1
x_hold[1, int(t_end / 10 + 1)] = -1
d_hold = np.zeros((n_out, t_end + 1))
for t in range(1, t_end + 1):
    z = np.concatenate((x_hold[:, t - 1], d_hold[:, t - 1]))
    d_hold[:, t] = M.dot(z)

# training
M = 0.02 * (np.random.random(M.shape) - .5)
# matrices of partial derivatives of out states
H = np.zeros((n_out, 2 + n_out, n_out))
for c in range(n_its):
    if np.random.rand() >= .5:
        x = x_hold
        d = d_hold
    else:
        x = -1 * x_hold
        d = -1 * d_hold
    z = np.concatenate((x[:, 0], [0, 0]))
    for t in range(1, t_end + 1):
        y = M.dot(z)
        e = d[:, t] - y
        H_pre = H
        H = np.zeros(H.shape)
        # update H for each out state
        for k in range(n_out):
            for l in range(n_out):
                hold = M[k, l + 2] * H_pre[:, :, l]
                hold[k, :] = hold[k, :] + z
                H[:, :, k] = H[:, :, k] + hold
        deltaM = np.zeros(M.shape)
        for k in range(n_out):
            deltaM = deltaM + e[k] * H[:, :, k]
        deltaM = lr * deltaM
        M = M + deltaM
        z = np.concatenate((x[:, t], y))

# check the trained

out = np.zeros(d_hold.shape)
z = np.concatenate((x_hold[:, 0], [0, 0]))
for t in range(1, t_end + 1):
    out[:, t] = M.dot(z)
    z = np.concatenate((x_hold[:, t], out[:, t]))

print('Trained weights:')
print(M)

t_vec = range(t_end + 1)
plt.plot(t_vec, d_hold[0, :], 'bx', label='actual up', linewidth=.5, markersize=1)
plt.plot(t_vec, d_hold[1, :], 'bx', label='actual down', linewidth=.5, markersize=1)
plt.plot(t_vec, out[0, :], 'r.-', label='calc up', linewidth=.5, markersize=0.3)
plt.plot(t_vec, out[1, :], 'r.-', label='calc down', linewidth=.5, markersize=0.3)
plt.legend()
plt.show()
