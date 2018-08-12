import numpy as np
from util import *
import matplotlib.pyplot as plt

pattern = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
input = pattern[np.newaxis, :]
for i in range(7):
    pattern = np.roll(pattern, 1)
    input = np.vstack((input, pattern))

input = input / np.linalg.norm(input, axis=1, keepdims=True)
output = kohonen_som(input, n_out=7, n_hood=5, n_its=100)

plt.imshow(output, cmap='gray')
plt.colorbar()
plt.show()
