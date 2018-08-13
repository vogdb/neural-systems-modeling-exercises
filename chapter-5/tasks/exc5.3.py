import numpy as np
import matplotlib.pyplot as plt


def get_x(tf):
    return np.exp(-np.abs(cf - tf) * qf)[np.newaxis, :]


qf = 3
in_dim = 20
cf = np.linspace(1, 2, in_dim)
n_out = 10
n_hood = 1
n_its = 500
n_in = 10
lr = 2
lr_dec = 0.99

W = np.random.random(size=(n_out, in_dim))
for i in range(n_its):
    tf = np.random.uniform(0, 1)
    train_in = get_x(tf)
    y = W.dot(train_in.T)
    max_y_idx = np.argmax(y)
    fn = max(max_y_idx - n_hood, 0)
    ln = min(max_y_idx + n_hood + 1, n_out)
    W[fn:ln] = W[fn:ln] + lr * train_in
    W[fn:ln] = W[fn:ln] / np.linalg.norm(W[fn:ln], axis=1, keepdims=True)
    lr = lr * lr_dec

tf_vec = np.linspace(1, 2, 10)
output = []
for tf in tf_vec:
    x = get_x(tf)
    # plt.plot(np.linspace(1, 2, 20), x)
    y = W.dot(x.T).T
    print(y.shape)
    plt.plot(tf_vec, np.squeeze(y))
    # plt.plot(tf_vec, W.dot(x))
    output.append(np.squeeze(y))
output = np.array(output)

print(output.shape)
# for y in output:
#     plt.plot(tf_vec, y, label='')
# plt.legend()
plt.show()
