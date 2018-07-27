from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=6, suppress=True)

time = 1000
n = 2

background = 10
x = np.ones((n, time)) * background
x[0, 101] += 1  # push input
x[1, 101] -= 1  # pull input

y = np.zeros((n, time))

w = np.array([
    [0.5, -0.5001],
    [-.5001, 0.5]
])
v = np.array([
    [1, 0],
    [0, 1]
])

for t in range(1, time):
    y[:, t] = w.dot(y[:, t - 1]) + v.dot(x[:, t - 1])

figure = plt.figure()
plt.plot(range(1, time), y[0, 1:], label='y1')
plt.plot(range(1, time), y[1, 1:], label='y2')
plt.legend()
plt.show()

# eig_val, eig_vec = np.linalg.eig(w)
# print(eig_val)
# print(eig_vec)
