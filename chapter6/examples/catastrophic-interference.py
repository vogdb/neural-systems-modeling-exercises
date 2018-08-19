import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
import matplotlib.pyplot as plt

n = 42
x = np.linspace(0, np.pi, n + 1)[:, np.newaxis]
x_odd = x[::2]
x_even = x[1::2]
y_odd = np.sin(x_odd)
y_even = np.exp(-x_even)

model = Sequential()
model.add(Dense(10, activation='sigmoid', input_dim=x.shape[1]))
model.add(Dense(1, activation='sigmoid'))
sgd = SGD(lr=.1)
model.compile(loss='mean_squared_error', optimizer=sgd)

model.fit(x_odd, y_odd, epochs=100000, verbose=0)

y_odd_predict1 = model.predict(x_odd)
# print(np.hstack((y_odd, y_odd_predict1)))
print(np.allclose(y_odd[1:-1], y_odd_predict1[1:-1], atol=0.05))

model.fit(x_even, y_even, epochs=100000, verbose=0)
y_even_predict = model.predict(x_even)
print(np.allclose(y_even[1:-1], y_even_predict[1:-1], atol=0.05))
y_odd_predict2 = model.predict(x_odd)
print(np.allclose(y_odd[1:-1], y_odd_predict2[1:-1], atol=0.05))

plt.plot(x_odd, y_odd_predict1, 'b.', label='y_odd_predict1')
plt.plot(x_even, y_even_predict, 'g-', label='y_even_predict')
plt.plot(x_odd, y_odd_predict2, 'r--', label='y_odd_predict2')
plt.legend()
plt.savefig('catastrophic-interference.png')
