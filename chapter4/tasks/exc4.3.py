from chapter4.tasks.util import *
import numpy as np
import matplotlib.pyplot as plt

p_mtrx = np.random.choice([0, 1], size=(4,20), p=[0.9, 0.1])
print(p_mtrx)
hb, pr, po, hp = build_connectivity_matrix(p_mtrx)

fig = plt.figure()

for idx, p in enumerate(p_mtrx):
    state = async_update(hb, p, 100)
    # state = async_update(hp, p, 100)
    ax_s = fig.add_subplot(23 * 10 + idx + 1)
    ax_s.imshow(state, cmap='gray')

plt.show()
