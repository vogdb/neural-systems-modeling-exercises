import numpy as np
from util import *
import matplotlib.pyplot as plt

input = np.array([
    [5, 0, 0],
    [4, 1, 0],
    [3, 1, 1],
    [0, 5, 0],
    [0, 4, 1],
    [1, 3, 1],
    [0, 0, 5],
    [0, 1, 4],
    [1, 1, 3],
])
input = input / np.linalg.norm(input, axis=1, keepdims=True)
print(input)
output = kohonen_som(input, n_out=6, n_hood=1, n_its=100)
print(output)

plt.imshow(output, cmap='gray')
plt.colorbar()
plt.show()
