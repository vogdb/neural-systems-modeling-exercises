import numpy as np
import matplotlib.pyplot as plt

time = 1000
x = np.zeros(time)
x[11] = 1

# number of units
n = 25
y = np.zeros((n, time))

w_self = np.diag(np.ones(n) * 0.95)
w_next = np.eye(n, k=-1) * 0.05
w = w_self + w_next
v = np.zeros(n)
v[0] = 1

for t in range(1, time):
    y[:, t] = w.dot(y[:, t - 1]) + v.dot(x[t - 1])


figure = plt.figure()
plt.plot(range(time), y[-2, :], label='y' + str(n-1))
plt.plot(range(time), y[-1, :], label='y' + str(n))
plt.legend()
plt.show()
