import numpy as np
import matplotlib.pyplot as plt

time = 100
x = np.array([0] * time)
x[11] = 1

y = np.zeros((2, time))

w11 = 0.95
w12 = 0
w21 = 0.5
w22 = 0.6
w = np.array([[w11, w12], [w21, w22]])

v1 = 1
v2 = 0
v = np.array([v1, v2])

for t in range(1, time):
    y[:, t] = w.dot(y[:, t - 1]) + v.dot(x[t - 1])

figure = plt.figure()
plt.plot(range(time), x, label='x')
plt.plot(range(time), y[0, :], label='y1')
plt.plot(range(time), y[1, :], label='y2')
plt.legend()
plt.show()
