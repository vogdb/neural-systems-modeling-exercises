import numpy as np


def kohonen_som(input, n_out=7, lr=1, lr_dec=1, n_hood=1, n_its=100):
    n_in, in_dim = input.shape
    W = np.random.random(size=(n_out, in_dim))
    for i in range(n_its):
        train_in = input[np.random.randint(0, n_in)]
        y = W.dot(train_in)
        max_y_idx = np.argmax(y)
        fn = max(max_y_idx - n_hood, 0)
        ln = min(max_y_idx + n_hood + 1, n_out)
        W[fn:ln] = W[fn:ln] + lr * train_in
        W[fn:ln] = W[fn:ln] / np.linalg.norm(W[fn:ln], axis=1, keepdims=True)
        lr = lr * lr_dec
    output = W.dot(input.T).T
    return output
