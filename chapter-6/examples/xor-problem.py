from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

import numpy as np

# x_train = np.array([
#     [0, 0, 1],
#     [1, 0, 1],
#     [0, 1, 1],
#     [1, 1, 1],
# ])
x_train = np.array([
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1],
])
y_train = np.array([0, 1, 1, 0])[:, np.newaxis]

model = Sequential()
model.add(Dense(1, activation='sigmoid', input_dim=x_train.shape[1], use_bias=False))

sgd = SGD(lr=1.)
model.compile(loss='mean_squared_error', optimizer=sgd)

model.fit(x_train, y_train, epochs=100000, verbose=0)

y_predict = model.predict(x_train)
print(y_predict)
W = model.layers[0].get_weights()
print(W)
