from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=5, suppress=True)

time = 500
x = np.zeros(time)
fly_t = 11
land_t = 351
x[fly_t:land_t] = 1

# number of units
n = 4
y = np.zeros((n, time))

w = np.array([
    [1, 3, 0, 0],
    [-12, 1, 6, 0],
    [0, -6, 1, -12],
    [0, 0, 3, 1],
])

v = np.array([0, 12, 11.9, 0])

for t in range(1, time):
    q = w.dot(y[:, t - 1]) + v.dot(x[t - 1])
    y[:, t] = 1 / (1 + np.exp(-q))

figure = plt.figure()
plt.plot(range(time), y[0, :], label='y1')
plt.plot(range(time), y[1, :], label='y2')
plt.plot(range(time), y[2, :], label='y3')
plt.plot(range(time), y[3, :], label='y4')
plt.legend()
plt.show()
