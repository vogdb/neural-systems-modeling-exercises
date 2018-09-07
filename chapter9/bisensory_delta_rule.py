from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

n_in = 2
n_out = 1
lr = .1
b = 1
n_its = 100000

# 1 - present, 0 - absent
e_v0, e_v1 = 2, 4
e_a0, e_a1 = 2, 3
sd_v0, sd_v1 = 1, 1
sd_a0, sd_a1 = 1, 1
# target absent/present correlation coefficient
r0, r1 = 0, 0
# prior of target
pt1 = .5
pt0 = 1 - pt1

# for random choosing of target
cum_prior = np.cumsum([pt0, pt1])
# correlation matrices
C1 = np.array([
    [sd_v1, 0],
    [sd_a1 * r1, sd_a1 * np.sqrt(1 - r1 ** 2)]
])
C0 = np.array([
    [sd_v0, 0],
    [sd_a0 * r0, sd_a0 * np.sqrt(1 - r0 ** 2)]
])

V = 2 * np.random.rand(n_out, n_in + 1) - 1

for c in range(n_its):
    choose = np.where(np.random.rand() < cum_prior)[0][0]
    if choose == 0:
        d = 0
        input_va = C0.dot(np.random.normal(size=(2, 1))) + np.array([e_v0, e_a0])[:, np.newaxis]
    else:
        d = 1
        input_va = C1.dot(np.random.normal(size=(2, 1))) + np.array([e_v1, e_a1])[:, np.newaxis]
    x = np.vstack((input_va, b * np.ones((1, input_va.shape[1]))))
    y = 1 / (1 + np.exp(-V.dot(x)))
    dy = y * (1 - y)
    err = d - y
    g = err * dy
    deltaV = lr * g * x.T
    V += deltaV

# value range for V and A
max_val = 9
val_num = 30
val_range = np.linspace(0, max_val, val_num)

input_va = np.vstack((val_range, val_range, b * np.ones(val_range.shape)))
input_vBGa = np.vstack((e_v0 * np.ones(val_range.shape), val_range, b * np.ones(val_range.shape)))
input_vaBG = np.vstack((val_range, e_a0 * np.ones(val_range.shape), b * np.ones(val_range.shape)))
input = np.hstack((input_va, input_vBGa, input_vaBG))
output = 1 / (1 + np.exp(-V.dot(input)))
output = output.flatten()
output_va = output[:val_num]
output_vBGa = output[val_num:2 * val_num]
output_vaBG = output[2 * val_num:]

if max(output_vaBG) > max(output_vBGa):
    MSE = 100 * (output_va - output_vaBG) / output_vaBG
else:
    MSE = 100 * (output_va - output_vBGa) / output_vBGa

plt.figure(figsize=(12, 8))
ax1 = plt.subplot(121)
ax1.set_title('NN response')
ax1.plot(val_range, output_va, 'ro', label='cross-modal')
ax1.plot(val_range, output_vaBG, 'g>', label='visual')
ax1.plot(val_range, output_vBGa, 'bs', label='auditory')
ax1.legend()

ax2 = plt.subplot(122)
ax2.set_title('MSE')
ax2.plot(val_range, MSE, 'black')

plt.show()
