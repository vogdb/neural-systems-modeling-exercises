import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
import matplotlib.pyplot as plt

n = 42
odd_idxs = np.s_[::2]
even_idxs = np.s_[1::2]
x = np.zeros((n + 1, 3))
x[:, 0] = np.linspace(0, np.pi, n + 1)
x[odd_idxs, 1] = 1
x[even_idxs, 2] = 1
y = np.zeros(x.shape[0])
y[odd_idxs] = np.sin(x[odd_idxs, 0])
y[even_idxs] = np.exp(-x[even_idxs, 0])

model = Sequential()
model.add(Dense(10, activation='sigmoid', input_dim=x.shape[1]))
model.add(Dense(1, activation='sigmoid'))
sgd = SGD(lr=.01)
model.compile(loss='mean_squared_error', optimizer=sgd)

model.fit(x, y, epochs=100000, verbose=0)

y_odd_predict = model.predict(x[odd_idxs, :])
y_even_predict = model.predict(x[even_idxs, :])

plt.plot(x[odd_idxs, 0], y_odd_predict, 'b.', label='odd sin predict')
plt.plot(x[even_idxs, 0], y_even_predict, 'g-', label='even exp(-x) predict')
plt.legend()
plt.savefig('exc6.3.png')
