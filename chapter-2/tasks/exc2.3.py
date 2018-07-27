import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=6, suppress=True)

time = 100
x = np.zeros(time)
x[11] = 1

# number of units
n = 2
y = np.zeros((n, time))

w = np.array([
    [0.9, -0.43585],
    [0.43585, 0.9],
])

# units in series, so only the first unit receives input
v = np.zeros(n)
v[0] = 1

for t in range(1, time):
    y[:, t] = w.dot(y[:, t - 1]) + v.dot(x[t - 1])

eig_val, eig_vec = np.linalg.eig(w)
print(eig_val)
print(np.abs(eig_val))
print(eig_vec)

figure = plt.figure()
plt.plot(range(time), y[-2, :], label='y' + str(n - 1))
plt.plot(range(time), y[-1, :], label='y' + str(n))
plt.legend()
plt.show()
