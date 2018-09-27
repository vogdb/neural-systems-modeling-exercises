from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

n_hid = 8
n_out = 2
n_in = 2
lr = 6
n_its = 10000
t_end_dk = 30  # decay end time
t_dk = np.arange(t_end_dk - 1)
t_end = 2 * t_end_dk
# canal time constant
tau_canal = 1
# VOR time constant
tau_vor = 4
xdk_canal = np.exp(-t_dk / tau_canal)
xdk_vor = np.exp(-t_dk / tau_vor)
canal_up = np.concatenate((
    0.5 * np.ones(5), 0.5 + .1 * xdk_canal[:t_end_dk - 5]
))
canal_down = np.concatenate((
    0.5 * np.ones(5), 0.5 - .1 * xdk_canal[:t_end_dk - 5]
))
vor_up = np.concatenate((
    0.5 * np.ones(7), 0.5 + .1 * xdk_vor[:t_end_dk - 7]
))
vor_down = np.concatenate((
    0.5 * np.ones(7), 0.5 - .1 * xdk_vor[:t_end_dk - 7]
))
x_hold = np.vstack((
    np.concatenate((canal_up, canal_down)),
    np.concatenate((canal_down, canal_up))
))
d_hold = np.vstack((
    np.concatenate((vor_down, vor_up)),
    np.concatenate((vor_up, vor_down))
))
M = 2 * (np.random.rand(n_hid + 2, n_hid + 4) - 0.5)
H = np.zeros((n_hid + 2, n_hid + 4, n_hid + 2))

msk = np.ones((n_hid + n_out, n_hid + n_out + n_in))
# output neurons do not receive inputs
msk[n_hid:n_hid + n_out, :n_out] = 0
# outputs do not project anywhere
msk[:, n_in + n_hid:n_in + n_hid + n_out] = 0
# hidden-output weights
msk[n_hid: n_hid + n_out, n_in: n_in + n_hid] = 0
# hidden same-side weights
n_hid_half = int(n_hid / 2)
msk[:n_hid_half, n_in: n_in + n_hid_half] = 0
msk[n_hid_half: n_hid, n_in + n_hid_half: n_in + n_hid] = 0

# print(msk)
M = M * msk
np.set_printoptions(2, suppress=True)
u_abs = 2 / n_hid
U = np.ones((1, n_hid_half)) * u_abs
U = np.vstack((
    np.hstack((-U, U)),
    np.hstack((U, -U))
))
# print(U)
M[n_hid:n_hid + 2, 2:n_hid + 2] = U
print('Initial weights:')
print(M)

# training
y = .5 * np.ones(n_hid + n_out)
for c in range(n_its):
    if np.random.rand() >= .5:
        x = x_hold
        d = d_hold
    else:
        x = x_hold[::-1, :]
        d = d_hold[::-1, :]

    z = np.concatenate((x[:, 0], y))
    for t in range(1, t_end):
        q = M.dot(z)
        y = 1 / (1 + np.exp(-q))
        e = d[:, t] - y[n_hid:n_hid + n_out]
        H_pre = H
        H = np.zeros(H.shape)
        # update H for each non-input state
        for k in range(n_hid + n_out):
            for l in range(n_hid + n_out):
                hold = M[k, l + n_in] * H_pre[:, :, l]
                hold[k, :] = hold[k, :] + z
                H[:, :, k] = H[:, :, k] + hold
            d_squash = y[k] * (1 - y[k])
            H[:, :, k] = d_squash * H[:, :, k]
        deltaM = np.zeros(M.shape)
        for k in range(n_hid, n_hid + n_out):
            deltaM = deltaM + e[k - n_hid] * H[:, :, k]
        deltaM = lr * deltaM * msk
        M = M + deltaM
        # eliminate positive hidden-hidden weights
        M[:n_hid, n_in:n_in + n_hid] = np.clip(M[:n_hid, n_in:n_in + n_hid], None, 0)
        z = np.concatenate((x[:, t], y))

print('Trained weights:')
print(M)

# trained response
out = np.zeros((len(y), t_end))
out[:, 0] = y
z = np.concatenate((x_hold[:, 0], y))
for t in range(1, t_end):
    q = M.dot(z)
    y = 1 / (1 + np.exp(-q))
    out[:, t] = y
    z = np.concatenate((x_hold[:, t], y))

# plotting
plt.figure(figsize=(14, 8))

t_vec = range(t_end)
ax1 = plt.subplot(311)
ax1.plot(t_vec, x_hold[0, :], 'k-.', label='LHC', linewidth=.5)
ax1.plot(t_vec, x_hold[1, :], 'k-.', label='RHC', linewidth=.5)
ax1.plot(t_vec, d_hold[0, :], 'b--', label='desired vor', linewidth=.5)
ax1.plot(t_vec, d_hold[1, :], 'b--', label='desired vor', linewidth=.5)
ax1.plot(t_vec, out[n_hid, :], 'r', label='calc vor', linewidth=.5)
ax1.plot(t_vec, out[n_hid + 1, :], 'r', label='calc vor', linewidth=.5)
ax1.legend()

# ax2 = plt.subplot(212)
# ax2.plot(t_vec, out[0, :], 'g', label='LVN')
# ax2.plot(t_vec, out[1, :], 'b--', label='RVN')
# ax2.legend()

ax2 = plt.subplot(312)
ax2.plot(t_vec, out[0, :], 'g', label='LVN 1')
ax2.plot(t_vec, out[2, :], 'b--', label='RVN 1')
ax2.legend()

ax3 = plt.subplot(313)
ax3.plot(t_vec, out[1, :], 'g', label='LVN 2')
ax3.plot(t_vec, out[3, :], 'b--', label='RVN 2')
ax3.legend()

plt.show()
