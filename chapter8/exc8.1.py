import numpy as np
from infocomp import infocomp

px1, px2 = .8, .7
# V = np.array([
#     [1, 1],
#     [1, 1],
# ])
# V = np.array([
#     [1, 0],
#     [0, 1],
# ])
V = np.array([
    [1, 1],
    [0, 1],
])

threshold = .7
input = np.array([
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1],
])
output = np.zeros(input.shape)
in_state_num = input.shape[0]
out_state_num = output.shape[0]
condi = np.zeros((out_state_num, in_state_num))

for l in range(in_state_num):
    x = input[l]
    y = V.dot(x)
    y = np.clip(y, 0, 1)
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
print(condi)

print('H_X: {:.2f}, H_Y: {:.2f}, I: {:.2f}'.format(hX, hY, I))
