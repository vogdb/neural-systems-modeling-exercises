from __future__ import division

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from util import laminate_array_to_matrix

nt = 20

x = np.ones(50)
x[17:36] = 3

p = np.zeros(50)
p[:3] = [-1, 2, -1]
p = np.roll(p, -1)
p = 2 * p

V = laminate_array_to_matrix(p)
n = V.shape[0]

y = V.dot(x)
y = 1 / (1 + np.exp(-y))

fig = plt.figure(figsize=(8, 8))

ax_V = fig.add_subplot(121, projection='3d')
V_x = np.arange(n)
V_y = np.arange(n)
V_x, V_y = np.meshgrid(V_y, V_y)
ax_V.set_title('Weights')
ax_V.plot_surface(V_x, V_y, V)

ax_y = fig.add_subplot(122)
ax_y.plot(y)
ax_y.set_xlabel('y unit')

plt.show()
