import numpy as np
from infocomp import infocomp

n_out = 1  # number of output units
px1 = .9
px0 = 1 - px1
py0x0 = .9
py1x0 = 1 - py0x0
py1x1 = .9
py0x1 = 1 - py1x1

n_y = np.power(2, n_out)
# conditional when x = 0
pyx0 = 1
# conditional when x = 1
pyx1 = 1
for i in range(n_out):
    pyx0 = np.kron([py0x0, py1x0], pyx0)
    pyx1 = np.kron([py0x1, py1x1], pyx1)
pyx0 = pyx0[:, np.newaxis]
pyx1 = pyx1[:, np.newaxis]
condi = np.hstack((pyx0, pyx1))

h_x, h_y, I = infocomp([px0, px0], condi)
print('H_X: {:.2f}, H_Y: {:.2f}, I: {:.2f}'.format(h_x, h_y, I))
