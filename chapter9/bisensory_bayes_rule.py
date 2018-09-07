from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

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
# value range for V and A
max_val = 9
val_num = 30
val_range = np.linspace(0, max_val, val_num)

visual, audio = np.meshgrid(val_range, val_range)
# 2D Gaussian likelihood of T = 1
D2pvat1 = (1 / (2 * np.pi * sd_v1 * sd_a1 * np.sqrt(1 - r1 ** 2))) * \
          np.exp(
              -(1 / (2 * (1 - r1 ** 2))) * (
                      ((visual - e_v1) / sd_v1) ** 2 -
                      2 * r1 * (((visual - e_v1) / sd_v1) * ((audio - e_a1) / sd_a1)) +
                      ((audio - e_a1) / sd_a1) ** 2
              )
          )
D2pvat1 = D2pvat1 / np.sum(D2pvat1)

# 2D Gaussian likelihood of T = 0
D2pvat0 = (1 / (2 * np.pi * sd_v0 * sd_a0 * np.sqrt(1 - r0 ** 2))) * \
          np.exp(
              -(1 / (2 * (1 - r0 ** 2))) * (
                      ((visual - e_v0) / sd_v0) ** 2 -
                      2 * r0 * (((visual - e_v0) / sd_v0) * ((audio - e_a0) / sd_a0)) +
                      ((audio - e_a0) / sd_a0) ** 2
              )
          )
D2pvat0 = D2pvat0 / np.sum(D2pvat0)

# evidence of V and A
D2pva = D2pvat0 * pt0 + D2pvat1 * pt1
# indices of SP(spontaneous or background) activity of sensory input neurons
bg_v_idx = np.where(val_range >= e_v0)[0][0]
bg_a_idx = np.where(val_range >= e_a0)[0][0]

# cross-modal cut likelihoods
pvat1 = np.diag(D2pvat1)
pvat0 = np.diag(D2pvat0)
# cross-modal cut evidence
pva = np.diag(D2pva)
# visual specific cut likelihoods
pvaBGt1 = D2pvat1[bg_a_idx, :]
pvaBGt0 = D2pvat0[bg_a_idx, :]
# visual specific cut evidence
pvaBG = D2pva[bg_a_idx, :]
# audio specific cut likelihoods
pvBGat1 = D2pvat1[:, bg_v_idx]
pvBGat0 = D2pvat0[:, bg_v_idx]
# audio specific cut evidence
pvBGa = D2pva[:, bg_v_idx]

# posterior
# cross-modal
pt1va = (pvat1 * pt1) / pva
# visual
pt1vaBG = (pvaBGt1 * pt1) / pvaBG
# audio
pt1vBGa = (pvBGat1 * pt1) / pvBGa

# MSE
if np.max(pt1vaBG) > np.max(pt1vBGa):
    MSE = 100 * (pt1va - pt1vaBG) / pt1vaBG
else:
    MSE = 100 * (pt1va - pt1vBGa) / pt1vBGa

plt.figure(figsize=(12, 8))
ax1 = plt.subplot(121)
ax1.set_title('visual and auditory input')
ax1.plot(val_range, pt1va, 'r', label='cross-modal')
ax1.plot(val_range, pt1vaBG, 'g', label='visual')
ax1.plot(val_range, pt1vBGa, 'b', label='auditory')
ax1.legend()

ax2 = plt.subplot(122)
ax2.set_title('MSE')
ax2.plot(val_range, MSE, 'black')

plt.show()
