import numpy as np

no_overlap_input = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
])

sparse_overlap_input = np.array([
    [1, 0, 0, 0],
    [1, 1, 0, 0],
    [1, 0, 1, 0],
    [0, 0, 0, 1],
])

#
# input = np.array([
#     [1, 0, 1, 0],
#     [1, 1, 1, 1],
#     [1, 1, 1, 0],
#     [1, 0, 0, 1],
# ])

output = np.array([
    [1],
    [1],
    [0],
    [0],
])


def train_plain_hebb(input, output):
    in_unit_num = input.shape[1]
    out_unit_num = output.shape[1]
    pattern_num = input.shape[0]
    W = np.zeros((out_unit_num, in_unit_num))

    for i in range(out_unit_num):
        for j in range(in_unit_num):
            for l in range(pattern_num):
                # hebb is in this multiplication
                W[i, j] += output[l][i] * input[l][j]

    return W


def train_smart_hebb(input, output):
    return output.T.dot(input)


def train_smart_hp(input, output):
    return (2 * output - 1).T.dot((2 * input - 1))


def calc_plain(W, input):
    pattern_num = input.shape[0]
    out_unit_num = W.shape[0]
    output = np.zeros((pattern_num, out_unit_num))
    for l in range(pattern_num):
        pattern = input[l]
        y = W.dot(pattern)
        output[l, :] = y.T
    return np.clip(output, 0, 1)


def calc_smart(W, input):
    output = W.dot(input.T).T
    return np.clip(output, 0, 1)


print('no overlap input:\n', no_overlap_input)
W = train_smart_hebb(no_overlap_input, output)
print('no overlap W hebb', W)
print('no overlap hebb calc:\n', calc_smart(W, no_overlap_input))
print('--------------------------')
print('sparse overlap input:\n', sparse_overlap_input)
W = train_smart_hebb(sparse_overlap_input, output)
print('sparse overlap W hebb', W)
print('sparse overlap hebb calc:\n', calc_smart(W, sparse_overlap_input))
print('--------------------------')
print('sparse overlap input:\n', sparse_overlap_input)
W = train_smart_hp(sparse_overlap_input, output)
print('sparse overlap W hp', W)
print('sparse overlap hp calc:\n', calc_smart(W, sparse_overlap_input))
