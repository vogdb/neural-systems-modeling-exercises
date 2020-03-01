'''
An attempt to implement the smart synapse mechanism from 14.2 Klemm(2000)

- had to reduce `delta_penalty` from 1 to 0.01. 1 is a very huge number. With `theta_threshold` = 1 it would
penalize on every cycle.
'''

import numpy as np
import matplotlib.pyplot as plt

np.seterr(all='raise')


def winner_take_all(units, beta_noise):
    divisor = np.sum(np.exp(beta_noise * units))
    # p - probabilities
    p = np.exp(beta_noise * units) / divisor
    # winner = np.where(p == p.max(axis=1, keepdims=True), p, 0)
    winner = np.where(p == p.max(), p, 0) / p.max()
    return winner.astype(int)


class SmartSynapses:

    def __init__(self, X_train, z_train, beta_noise=10, delta_penalty=0.01, theta_threshold=2):
        self.X_train = X_train
        self.z_train = z_train

        self.beta_noise = beta_noise
        self.delta_penalty = delta_penalty
        self.theta_threshold = theta_threshold

        self.hidden_weights = np.random.randn(3, 3)
        self.hidden_memory = np.zeros(self.hidden_weights.shape)
        self.output_weights = np.random.randn(2, 3)
        self.output_memory = np.zeros(self.output_weights.shape)

    def random_weight_increment(self):
        w_random_increment = 0.001
        hidden_weights_increment = w_random_increment * np.random.rand(*self.hidden_weights.shape)
        output_weights_increment = w_random_increment * np.random.rand(*self.output_weights.shape)
        self.hidden_weights += hidden_weights_increment
        self.output_weights += output_weights_increment

    def train(self):
        self.train_errors = []
        sample_size = len(self.z_train)
        for epoch in range(1000):
            self.random_weight_increment()

            sample_index = np.random.randint(0, sample_size)
            sample_input = self.X_train[sample_index]
            sample_output = self.z_train[sample_index]
            hidden_q = self.hidden_weights.dot(sample_input)
            output_q = self.output_weights.dot(hidden_q)
            hidden_winner = winner_take_all(hidden_q, self.beta_noise)
            output_winner = winner_take_all(output_q, self.beta_noise)

            # r - reward
            r = 1 if np.array_equal(sample_output, output_winner) else -1

            input_indexes = np.nonzero(sample_input)[0]
            hidden_winner_indexes = np.nonzero(hidden_winner)[0]
            output_winner_indexes = np.nonzero(output_winner)[0]
            hidden_hebb_indexes = np.s_[hidden_winner_indexes, input_indexes]
            output_hebb_indexes = np.s_[output_winner_indexes, hidden_winner_indexes]

            self.hidden_memory[hidden_hebb_indexes] -= r
            self.output_memory[output_hebb_indexes] -= r
            self.hidden_memory = np.clip(self.hidden_memory, 0, self.theta_threshold)
            self.output_memory = np.clip(self.output_memory, 0, self.theta_threshold)

            hidden_penalty_indexes = np.nonzero(self.hidden_memory == self.theta_threshold)
            self.hidden_memory[hidden_penalty_indexes] = 0
            self.hidden_weights[hidden_penalty_indexes] -= self.delta_penalty

            output_penalty_indexes = np.nonzero(self.output_memory == self.theta_threshold)
            self.output_memory[output_penalty_indexes] = 0
            self.output_weights[output_penalty_indexes] -= self.delta_penalty
            self.train_errors.append(self.error(self.X_train, self.z_train))

    def predict(self, X):
        hidden = self.hidden_weights.dot(X.T)
        output = self.output_weights.dot(hidden)
        return output.T

    def error(self, X, z_true):
        z_predict = self.predict(X)
        return np.sum(np.abs(z_predict - z_true))


if __name__ == "__main__":
    X_xor = np.array([
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ])
    z_xor = np.array([
        [0, 1],
        [1, 0],
        [1, 0],
        [0, 1],
    ])
    smart_syns = SmartSynapses(X_xor, z_xor)
    smart_syns.train()
    plt.plot(smart_syns.train_errors)
    plt.show()
