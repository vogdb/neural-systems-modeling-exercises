import numpy as np
from sklearn.preprocessing import normalize
from scipy.optimize import fsolve
from infocomp import infocomp

n_x = 20  # number of input states
stf = 1  # input spatial field
n_in = stf * 2 + n_x  # number of input units
n_out = 30  # number of output units and states simultaneously
bg = .9  # background noise
n_hood = 10  # SOM neighbour size
lr = 1
n_its = 1000
V = np.random.rand(n_out, n_in)
V = normalize(V, norm='l2', axis=1)
# V = V / np.linalg.norm(V, axis=1, keepdims=True)

for c in range(n_its):
    x = np.ones((n_in, 1)) * bg
    r_tpos = np.random.randint(n_x)
    x[r_tpos:r_tpos + 2 * stf + 1] = 1
    y = V.dot(x)
    max_y_idx = np.argmax(y)
    fn = max(max_y_idx - n_hood, 0)
    ln = min(max_y_idx + n_hood + 1, n_out)
    V[fn:ln] = V[fn:ln] + lr * x.T
    V[fn:ln] = normalize(V[fn:ln], norm='l2', axis=1)

condi = np.zeros((n_out, n_x))
for i in range(n_x):
    x = np.ones((n_in, 1)) * bg
    x[i:i + 2 * stf + 1] = 1
    y = V.dot(x)
    y = y == np.max(y)
    y = y.astype(float).flatten()
    condi[:, i] = y / sum(y)

p_x = np.ones(n_x) / n_x
h_x, h_y, I = infocomp(p_x, condi)

d = fsolve(
    lambda d: np.log2(n_x) + d * np.log2(d) + (1 - d) * np.log2(1 - d) - d * np.log2(n_x - 1) - I,
    np.array([.1]),
    factor=0.5
)

print(condi)
used_out_states = np.argmax(condi, axis=0)
input_output_map = np.vstack((
    range(n_x),
    used_out_states
))
print(input_output_map)
print('{} input states are coded by {} output states'.format(n_x, len(set(used_out_states))))

print('H_X: {:.2f}, H_Y: {:.2f}, I: {:.2f}, d: {:.3f}'.format(h_x, h_y, I, d[0]))
