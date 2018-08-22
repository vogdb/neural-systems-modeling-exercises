import numpy as np
from scipy.special import expit

np.set_printoptions(precision=3, suppress=True)


def train(X, y, n_hid_num=1, epoch_num=100000, tolerance=.1, p=0.01):
    def predict(H, W):
        hidden = expit(H.dot(X.T))
        # add bias stubs to hidden
        hidden = np.vstack((hidden, np.ones((1, hidden.shape[1]))))
        output = expit(W.dot(hidden))
        return output.T

    def calc_error(y_predict):
        return np.sum(np.abs(y - y_predict))

    pattern_num, in_unit_num = X.shape
    out_unit_num = y.shape[1]

    # + 1 extra columns are for bias
    W = 2 * np.random.rand(out_unit_num, n_hid_num + 1) - 1
    H = 2 * np.random.rand(n_hid_num, in_unit_num + 1) - 1

    # add bias stubs to X
    X = np.hstack((X, np.ones((X.shape[0], 1))))

    error = calc_error(predict(H, W))
    epoch = 0
    while error > tolerance:
        perturbation_H = p * np.random.normal(size=H.shape)
        H_perturbed = H + perturbation_H
        perturbation_W = p * np.random.normal(size=W.shape)
        W_perturbed = W + perturbation_W
        error_new = calc_error(predict(H_perturbed, W_perturbed))
        if error_new < error:
            W = W_perturbed
            H = H_perturbed
            error = error_new
        epoch += 1
        if epoch > epoch_num:
            break
    return predict(H, W), H, W, epoch


if __name__ == "__main__":
    X = np.array([
        [0, 0],
        [1, 0],
        [0, 1],
    ])
    y = np.array([0, 1, 1])[:, np.newaxis]

    y_predict, H, W, epoch = train(X, y, n_hid_num=1000)
    print(epoch)
    print(np.hstack((y, y_predict)))
    print(H)
    print(W)
    # n_hid_num = 10 => epoch = 2727, 3268, 3295, 2185...
    # n_hid_num = 100 => epoch = 1496, 1375, 1470, 1948...
    # n_hid_num = 1000 => epoch = 603, 100001 (incorrect), 593, 501, 444, 100001(incorrect)...