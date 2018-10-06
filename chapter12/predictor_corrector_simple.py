import numpy as np

np.set_printoptions(precision=2)

n = 7

# Conditional T(t) given T(t-1), A(t-1) = 0
profileT_gT_A0 = np.array([.2, .4, .2, .1, 0, 0, .1])
prbT_gT_A0 = np.zeros((n, n))
for i in range(n):
    prbT_gT_A0[:, i] = profileT_gT_A0
    profileT_gT_A0 = np.roll(profileT_gT_A0, 1)

# Conditional V(t) given T(t)
profileV_gT = np.array([.8, .1, 0, 0, 0, 0, .1])
prbV_gT = np.zeros((n, n))
for i in range(n):
    prbV_gT[:, i] = profileV_gT
    profileV_gT = np.roll(profileV_gT, 1)

print(prbV_gT)

# previous estimate T = 0 (center fovea position)
prevT_est = [0, 0, 0, 1, 0, 0, 0]

# current prior T, assume that saccade didn't happen, A(t-1) = 0
currT_prior = prbT_gT_A0.dot(prevT_est)
# currT_prior [0. 0. 0.1 0.2 0.4 0.2 0.1]

# let the observation be +2 => V = position 6
currV_gT = prbV_gT[6 - 1, :]
# currV_gT [0. 0. 0. 0. 0.1 0.8 0.1]

currV_evid = currV_gT.dot(currT_prior)
# V_evidence 0.21

currT_gV = (currV_gT / currV_evid) * currT_prior
# currT_gV [0. 0. 0. 0. 0.19 0.76 0.05]
