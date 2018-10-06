import numpy as np

np.set_printoptions(precision=2)

n = 80
space = np.arange(-n, n + 1)
n_space = len(space)
shift = int((n_space + 1) / 2)
cell_pref_t = 10
# enter light off position
light_off = 12
# enter light on position
light_on = 21
# number of positions(spaces) target will assume
n_tpos = 31
# prediction variance
prbT_gT_var = 25
# observation variance
prbV_gT_on_var = 9

# Conditional T(t) given T(t-1), A(t-1) = 0
profileT_gT_A0 = np.exp(-.5 * (space ** 2 / prbT_gT_var))
profileT_gT_A0 = profileT_gT_A0 / np.sum(profileT_gT_A0)
profileT_gT_A0 = np.roll(profileT_gT_A0, shift + 1)
prbT_gT_A0 = np.zeros((n_space, n_space))
for i in range(n_space):
    prbT_gT_A0[:, i] = profileT_gT_A0
    profileT_gT_A0 = np.roll(profileT_gT_A0, 1)

# Conditional V(t) given T(t) when light is off
prbV_gT_off = np.ones(n_space) / n_space
# Conditional V(t) given T(t) when light is on
profileV_gT_on = np.exp(-.5 * (space ** 2 / prbV_gT_on_var))
profileV_gT_on = profileV_gT_on / np.sum(profileV_gT_on)
profileV_gT_on = np.roll(profileV_gT_on, shift)
prbV_gT_on = np.zeros((n_space, n_space))
for i in range(n_space):
    prbV_gT_on[:, i] = profileV_gT_on
    profileV_gT_on = np.roll(profileV_gT_on, 1)
