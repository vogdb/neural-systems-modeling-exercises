from util import *
import numpy as np
import matplotlib.pyplot as plt

p_mtrx = np.zeros((3, 50))
p_mtrx[0, 0::2] = 1
p_mtrx[1].reshape(10, 5)[0::2, :] = 1
p_mtrx[2, 1::2] = 1

hp = build_connectivity_matrix_asym(p_mtrx)

fig = plt.figure()

p0 = np.ones(50)
p0.reshape(2, 25)[0] = 0
state = async_update(hp, p0, 800)
plt.imshow(state, cmap='gray')

plt.show()
