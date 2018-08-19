import numpy as np
from scipy.special import expit


def train(X, y, epoch_num=100000, tolerance=.1, p=0.01):
    def predict(W):
        return expit(W.dot(X.T).T)

    def calc_error(y_predict):
        return np.sum(np.abs(y - y_predict))

    pattern_num, in_unit_num = X.shape
    out_unit_num = y.shape[1]
    # W_dense = np.array([-1, -1, 1, 1])[np.newaxis, :]
    # W = W_dense
    W = 2 * np.random.rand(out_unit_num, in_unit_num) - 1

    error = calc_error(predict(W))
    epoch = 0
    while error > tolerance:
        # perturbation = p * np.random.choice([-1, 1], size=W.shape)
        perturbation = p * np.random.normal(size=W.shape)
        W_perturbed = W + perturbation
        error_new = calc_error(predict(W_perturbed))
        if error_new < error:
            W = W_perturbed
            error = error_new
        epoch += 1
        if epoch > epoch_num:
            break
    return predict(W), W, epoch


if __name__ == "__main__":
    X_sparse = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])
    X_dense = np.array([
        [1, 0, 1, 0],
        [1, 1, 1, 1],
        [1, 1, 1, 0],
        [1, 0, 0, 1],
    ])

    y_train = np.array([1, 1, 0, 0])[:, np.newaxis]

    y_predict, W, epoch = train(X_dense, y_train)
    print(epoch)
    print(y_predict)
    print(W)
