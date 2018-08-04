from __future__ import division

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


def generate_input(start, stop, distract=False, noise=False):
    input = np.zeros(51)
    input[start:stop] = 2
    if distract:
        input[start - 10: start - 8] = 1
        input[start - 5: start - 3] = 1
        input[stop + 3: stop + 5] = 1
        input[stop + 8: stop + 10] = 1
    if noise:
        input += np.random.normal(size=len(input))
    return input


def laminate_array_to_matrix(array):
    n = len(array)
    matrix = np.zeros((n, n))
    for i in range(n):
        matrix[i, :] = np.roll(array, i)
    return matrix


def gauss_pro(s=np.arange(-25, 26), sd=1):
    return np.exp(-.5 * (s / sd) ** 2)


nt = 20
rate = 1
cut = 0
sat = 10

x = generate_input(25, 27, distract=True, noise=False)
# x = generate_input(2, 4, distract=True, noise=False)
# x = generate_input(39, 41, distract=False, noise=True)

d = gauss_pro(sd=15)
g = gauss_pro(sd=3)
p = g - 0.3 * d
# p = g - 0.1 * d
p = np.roll(p, int(len(p) / 2) + 1)

W = laminate_array_to_matrix(p)
n = W.shape[0]

V = np.eye(n)
y = np.zeros((n, nt))

for t_i in range(1, nt):
    y[:, t_i] = (rate * W).dot(y[:, t_i - 1]) + V.dot(x)
    y[:, t_i] = np.maximum(y[:, t_i], cut)
    y[:, t_i] = np.minimum(y[:, t_i], sat)

fig = plt.figure(figsize=(8, 8))

ax_W = fig.add_subplot(121, projection='3d')
W_x = np.arange(n)
W_y = np.arange(n)
W_x, W_y = np.meshgrid(W_y, W_y)
ax_W.set_title('Weights')
ax_W.plot_surface(W_x, W_y, W)

ax_y = fig.add_subplot(122, projection='3d')
y_n = np.arange(n)
y_t = np.arange(nt)
y_t, y_n = np.meshgrid(y_t, y_n)
ax_y.plot_surface(y_t, y_n, y)
ax_y.set_xlabel('T')
ax_y.set_ylabel('y unit')
ax_y.set_title('y response')

plt.show()
