import numpy as np
import matplotlib.pyplot as plt

train_num = 20000
lr = .01

f = .5
t = np.linspace(0, (1 / f) * 10, train_num)
x_original = np.zeros((2, train_num))
x_original[0, :] = .5 * np.sin(2 * np.pi * f * t)
x_original[1, :] = np.random.rand(train_num) - .5
mix = np.array([
    [.6, .4],
    [.5, .5],
])
x_train = mix.dot(x_original)

V = np.array([
    [.03, .02],
    [.01, .04],
])
b = np.array([
    [.02],
    [.03],
])

batch_size = 50
del_V = np.zeros(V.shape)
del_b = np.zeros(b.shape)
for c in range(train_num):
    x = x_train[:, c][:, np.newaxis]
    q = V.dot(x) + b
    y = 1. / (1 + np.exp(-q))
    if c % batch_size == 0:
        V += del_V
        V[V < 0] = 0
        b += del_b
        del_V = np.zeros(V.shape)
        del_b = np.zeros(b.shape)
    else:
        del_V += lr * (np.linalg.inv(V).T + (1 - 2 * y).dot(x.T))
        del_b += lr * (1 - 2 * y)

y = np.empty_like(x_train)
for c in range(train_num):
    x = x_train[:, c][:, np.newaxis]
    q = V.dot(x) + b
    y[:, c] = (1. / (1 + np.exp(-q))).flatten()

ax_original1 = plt.subplot(221)
ax_original1.set_title('original')
ax_original1.plot(x_original[0])
ax_original2 = plt.subplot(222)
ax_original2.set_title('original')
ax_original2.plot(x_original[1])

ax_restored1 = plt.subplot(223)
ax_restored1.set_title('restored')
ax_restored1.plot(y[0])
ax_restored2 = plt.subplot(224)
ax_restored2.set_title('restored')
ax_restored2.plot(y[1])

plt.savefig('exc8.2.svg')
