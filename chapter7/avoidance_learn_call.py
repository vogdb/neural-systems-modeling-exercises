import numpy as np
import matplotlib.pyplot as plt

trial_num = 2000
pretrain_num = trial_num / 10
lr = 0.005
lr_prob = 0.005
w_sh = 0
w_fh = 1
w_ch = 1
hear = 1

call_rec = []
sumo_rec = []
fumo_rec = []
spin_rec = []

for c in range(1, trial_num):
    if c <= pretrain_num:
        rew_s, rew_f = 0, 1
    else:
        rew_s, rew_f = 1, 0
    call = 0 #w_ch * hear
    sumo = w_sh * hear
    fumo = w_fh * hear
    spin = int(sumo > fumo)
    prob = lr_prob + lr_prob * call
    if prob > np.random.rand():
        spin = 1 - spin
    call_rec.append(call)
    sumo_rec.append(sumo)
    fumo_rec.append(fumo)
    spin_rec.append(spin)
    pain = 0
    if c <= pretrain_num:
        pain = 0
    elif spin == 1:
        pain = 0
    elif spin == 0:
        pain = 1
    w_ch = w_ch + lr * (pain - w_ch)
    if spin == 1:
        w_sh = w_sh + lr * (rew_s - w_sh)
    else:
        w_fh = w_fh + lr * (rew_f - w_fh)

plt.plot(call_rec, label='call')
plt.plot(fumo_rec, label='fumo')
plt.plot(sumo_rec, label='sumo')
plt.plot(spin_rec, '.', label='spin')
plt.legend()
plt.savefig('avoidance_learn_call.svg')
