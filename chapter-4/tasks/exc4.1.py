from util import *
import matplotlib.pyplot as plt

p_mtrx = np.array([
    [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
])
hb, pr, po, hp = build_connectivity_matrix(p_mtrx)
# p0 = p_mtrx[1, :]
# p0 = [.1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
p0 = [.4, .3, .5, .3, .4, .2, .1, .1, .2, .1]
# state = sync_update(hp, p0, 10)
state = async_update(hp, p0, 100)

fig = plt.figure()
plt.imshow(state, cmap='gray')
plt.colorbar()
plt.show()
