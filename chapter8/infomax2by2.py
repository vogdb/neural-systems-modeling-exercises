import numpy as np
from chapter8.infocomp import infocomp

px1, px2 = .5, .5
train_num = 3000
lr = .01
threshold = .51
x_train = np.zeros((2, train_num))
x_train[0, :] = np.random.rand(train_num) < px1
x_train[1, :] = np.random.rand(train_num) < px2
input = np.array([
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1],
])
V = np.random.rand(2, 2)
b = np.random.rand(2, 1)

for c in range(train_num):
    x = x_train[:, c][:, np.newaxis]
    q = V.dot(x) + b
    y = 1. / (1 + np.exp(-q))
    del_V = lr * (np.linalg.inv(V).T + (1 - 2 * y).dot(x.T))
    del_b = lr * (1 - 2 * y)
    V += del_V
    V[V < 0] = 0
    b += del_b

output = np.zeros(input.shape)
in_state_num = input.shape[0]
out_state_num = output.shape[0]
condi = np.zeros((out_state_num, in_state_num))

for l in range(in_state_num):
    x = input[l][:, np.newaxis]
    q = V.dot(x) + b
    y = 1 / (1 + np.exp(-q))
    y = y > threshold
    y = np.asarray(y, dtype=int).flatten()
    if np.array_equal(y, [0, 0]):
        condi[0, l] = 1
    elif np.array_equal(y, [1, 0]):
        condi[1, l] = 1
    elif np.array_equal(y, [0, 1]):
        condi[2, l] = 1
    elif np.array_equal(y, [1, 1]):
        condi[3, l] = 1
    output[l] = y

pX = np.zeros(in_state_num)
pX[0] = (1 - px1) * (1 - px2)
pX[1] = px1 * (1 - px2)
pX[2] = (1 - px1) * px2
pX[3] = px1 * px2

hX, hY, I = infocomp(pX, condi)

print(input)
print(output)
print(V)
print(condi)

print('H_X: {:.2f}, H_Y: {:.2f}, I: {:.2f}'.format(hX, hY, I))
