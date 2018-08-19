import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import matplotlib.pyplot as plt
import numpy as np

x_train = np.array([
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1],
])
y_train = np.array([0, 1, 1, 0])[:, np.newaxis]


def train(n_hid, x, y):
    stop_monitor = keras.callbacks.EarlyStopping(monitor='loss', min_delta=1e-6, patience=10, verbose=0)
    model = Sequential()
    model.add(Dense(n_hid, activation='sigmoid', input_dim=x.shape[1]))
    model.add(Dense(1, activation='sigmoid'))

    sgd = SGD(lr=1.)
    model.compile(loss='mean_squared_error', optimizer=sgd)

    model.fit(x, y, epochs=10000, verbose=0, callbacks=[stop_monitor])

    # y_predict = model.predict(x)
    # print(y_predict)
    return stop_monitor.stopped_epoch


n_hid_list = range(2, 103, 10)
epoch_list = []
for n_hid in n_hid_list:
    epoch_list.append(train(n_hid, x_train, y_train))

plt.plot(n_hid_list, epoch_list)
plt.savefig('exc6.2.png')
