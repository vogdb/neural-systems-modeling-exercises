from chapter12.predict_correct_setup import *

n_out = 1
n_in = n_space + 2
lr = 0.1
b = 1
n_its = 20000
# saccade start position
sacc_start = 40
# light flip interval for training
light_switch_nt = 200
# is light on?
light_flip = 1

# weights
M = 0.2 * (np.random.rand(n_out, n_in + n_out) - 0.5)
T_pos_record = np.zeros(n_its)
SSC_record = np.zeros(n_its)
# input from SSC
x_SSC = np.zeros(n_space)
# initial values
T_pos = 0
SSC = 0
DSC = 0
y = 0
d = 0
z = np.array([b] + x_SSC + [DSC, y])

for t in range(n_its):
    T_pos_record[t] = T_pos
    SSC_record[t] = SSC
    q = M.dot(z)
    y = 1 / (1 + np.exp(-q))
    e = d - y
    d_squash = y * (1 - y)
    delta_M = lr * e * d_squash * z
    M += delta_M
    if t % light_switch_nt == 0:
        light_flip = 1 - light_flip
    if T_pos >= sacc_start:
        # saccade reset
        T_pos = 0
        SSC = 0
        DSC = 1
        x_SSC = np.zeros(n_space)
    else:
        # advance target according to P(T(t)|T(t-1), A(t-1)=0)
        prb_T_gT = prbT_gT_A0[:, T_pos + shift]
        # TODO, too boring for now...

#
# plt.figure(figsize=(14, 10))
#
# ax1 = plt.subplot(121)
# ax1.plot(pos_actual_on, 'b|', label='actual')
# ax1.plot(pos_estimate_on, 'g_', label='estimate')
# ax1.legend()
#
# ax2 = plt.subplot(122)
# ax2.plot(pbn_cell_on, 'b', label='light on')
# ax2.plot(pbn_cell_off, 'g', label='light off')
# ax2.legend()
#
# plt.show()
