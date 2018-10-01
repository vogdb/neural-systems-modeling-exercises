import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

lr = 0.3
n_its = 200
n_time = 50

x = np.zeros(n_time)
y = np.zeros(n_time)
v = np.zeros(n_time)
r = np.zeros(n_time)
z = np.zeros(n_time)

t_course = np.zeros((n_its, n_time))

# cue time
q_time = 10
# reward time
r_time = 30
x[q_time:r_time] = 1

for c in range(n_its):
    # if c == int(n_its / 2):
    # if c == int(n_its / 10):
    #     r[r_time] = 0
    # else:
    #     r[r_time] = 1
    r_jitter = r_time - np.random.randint(4)
    r[r_jitter] = 1
    y = np.concatenate(([0], np.diff(v * x)))
    z = y + r
    v = v + lr * x * np.concatenate((z[1:], [0]))
    t_course[c, :] = z

X = np.arange(0, n_time)
Y = np.arange(0, n_its)
X, Y = np.meshgrid(X, Y)

fig = plt.figure(figsize=(14,10))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, t_course, cmap='plasma')
ax.set_xlabel('time')
ax.set_ylabel('trial')
plt.show()
