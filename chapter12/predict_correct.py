import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=2)

n = 80
space = np.arange(-n, n + 1)
n_space = len(space)
shift = int((n_space + 1) / 2)
cell_pref_t = 10 + shift
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


def track_light_on():
    # target that is always on
    T_pos_actual = np.zeros(n_tpos)
    T_pos_estimate = np.zeros(n_tpos)
    pbn_cell = np.zeros(n_tpos)
    # initial T position
    T_pos = 0
    # initial posterior P(T(t-1)|V(t-1))
    prbT_gV = np.ones(n_space) / n_space
    for t in range(1, n_tpos):
        # prior of T, prediction
        prbT_prior = prbT_gT_A0.dot(prbT_gV)
        # likelihood of V given T of current position, P(V(t)|T(t))
        prbV_gT = prbV_gT_on[:, T_pos + shift]
        # evidence of V, P(V)
        prbV = prbV_gT.dot(prbT_prior)
        # estimate or current posterior, P(T(t)|V(t))
        prbT_gV = (prbV_gT / prbV) * prbT_prior
        # T estimate
        T_est = np.argmax(prbT_gV)
        T_est = T_est - shift
        # calc PBN response
        prbT_gV_cell_shift = np.roll(prbT_gV, -cell_pref_t)
        pbn_cell[t] = np.sum(prbT_gV_cell_shift[:int(n_space / 2 - 1)])
        # record for later
        T_pos_actual[t] = T_pos
        T_pos_estimate[t] = T_est
        T_pos += 1

    return T_pos_actual, T_pos_estimate, pbn_cell


def track_light_off():
    # target that is always on
    T_pos_actual = np.zeros(n_tpos)
    T_pos_estimate = np.zeros(n_tpos)
    pbn_cell = np.zeros(n_tpos)
    # initial T position
    T_pos = 0
    # initial posterior P(T(t-1)|V(t-1))
    prbT_gV = np.ones(n_space) / n_space
    for t in range(1, n_tpos):
        # prior of T, prediction
        prbT_prior = prbT_gT_A0.dot(prbT_gV)
        if light_off <= t <= light_on:
            prbV_gT = prbV_gT_off
        else:
            # likelihood of V given T of current position, P(V(t)|T(t))
            prbV_gT = prbV_gT_on[:, T_pos + shift]
        # evidence of V, P(V)
        prbV = prbV_gT.dot(prbT_prior)
        # estimate or current posterior, P(T(t)|V(t))
        prbT_gV = (prbV_gT / prbV) * prbT_prior
        # T estimate
        T_est = np.argmax(prbT_gV)
        T_est = T_est - shift
        # calc PBN response
        prbT_gV_cell_shift = np.roll(prbT_gV, -cell_pref_t)
        pbn_cell[t] = np.sum(prbT_gV_cell_shift[:int(n_space / 2 - 1)])
        # record for later
        T_pos_actual[t] = T_pos
        T_pos_estimate[t] = T_est
        T_pos += 1

    return T_pos_actual, T_pos_estimate, pbn_cell


pos_actual_on, pos_estimate_on, pbn_cell_on = track_light_on()
pos_actual_off, pos_estimate_off, pbn_cell_off = track_light_off()

plt.figure(figsize=(14, 10))

ax1 = plt.subplot(121)
ax1.plot(pos_actual_on, 'b|', label='actual')
ax1.plot(pos_estimate_on, 'g_', label='estimate')
ax1.legend()

ax2 = plt.subplot(122)
ax2.plot(pbn_cell_on, 'b', label='light on')
ax2.plot(pbn_cell_off, 'g', label='light off')
ax2.legend()

plt.show()
