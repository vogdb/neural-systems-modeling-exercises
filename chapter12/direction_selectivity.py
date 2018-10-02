import numpy as np
import matplotlib.pyplot as plt


def generate_input(n_ex, speed, dir_flag):
    input = np.eye(n_ex)
    if speed > 1:
        # discrete speed only
        speed = int(speed)
        input = input[::speed]
        # fill the rest with zeros
        fill_shape = (n_ex - input.shape[0], input.shape[1])
        input = np.vstack((input, np.zeros(fill_shape)))
    elif speed < 1:
        speed_reciprocal = int(1. / speed)
        input_slow = np.zeros(input.shape)
        for i in range(int(speed * input.shape[0])):
            input_slow[i * speed_reciprocal] = input[i]
        input = input_slow
    if dir_flag == 0:
        input = np.fliplr(input)
    return input


n_ex = 30
n_in = n_ex - 1
n_y = n_ex - 1
dir_flag = 1
speed = 2

input = generate_input(n_ex, speed, dir_flag)

# from excitatory x to inhibitory x
V_ex_in = np.hstack((np.zeros((n_in, 1)), np.eye(n_in)))
# from all x to y
# V_x_y = np.hstack((np.eye(n_y), np.zeros((n_y, 1)), -1 * np.eye(n_y)))
V_x_y = np.hstack((np.eye(n_y), np.zeros((n_y, 1)), -1 * np.triu(np.ones(n_y))))
U_y_z = np.ones((1, n_y))

x_ex = np.zeros(n_ex)
x_in = np.zeros(n_in)
y = np.zeros(n_y)
z_t = np.zeros(n_ex)
for t in range(n_ex):
    x_ex = input[t, :]
    x = np.concatenate((x_ex, x_in))
    q = V_x_y.dot(x)
    y = q > 0
    z_t[t] = U_y_z.dot(y)
    x_in = V_ex_in.dot(x_ex)

# plot active x units
ax_x = plt.subplot(211)
active_x_idx = {}
for x_idx, x_t in enumerate(input):
    x_a = np.nonzero(x_t)[0]
    if len(x_a) > 0:
        x_a = x_a[0]
        active_x_idx[x_idx] = x_a
ax_x.plot(active_x_idx.keys(), active_x_idx.values(), 'bs')
ax_x.set_ylim(0, n_ex)
ax_x.set_xlim(-1, n_ex + 1)

# plot z output
ax_z = plt.subplot(212)
ax_z.plot(range(n_ex), z_t, 'o')
ax_z.set_ylim(0, 2)
ax_z.set_xlim(-1, n_ex + 1)
plt.show()
