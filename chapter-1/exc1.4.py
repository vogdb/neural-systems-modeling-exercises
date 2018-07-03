import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

time = 100
num_of_inputs = 10
x = np.zeros((num_of_inputs, time))
# all inputs pulse at the same time
x[:, 11] = 1

num_of_neurons = 10
y = np.zeros((num_of_neurons, time))

w_input = np.random.randn(num_of_neurons, num_of_inputs)
w_feedback = np.random.randn(num_of_neurons, num_of_neurons)

cut = 0
saturation = 1000

for t in range(1, time):
    y[:, t] = w_input.dot(x[:, t]) + w_feedback.dot(y[:, t - 1])
    y[:, t] = np.clip(y[:, t], cut, saturation)


fig = plt.figure()

ax1 = fig.add_subplot(221, projection='3d')
ax1.set_title('Input weights')
X = np.arange(0, w_input.shape[0])
Y = np.arange(0, w_input.shape[1])
X, Y = np.meshgrid(X, Y)
ax1.set_xlabel('Neuron number')
ax1.set_ylabel('Input number')
ax1.set_zlabel('W')
ax1.scatter(X, Y, w_input)

ax2 = fig.add_subplot(223, projection='3d')
ax2.set_title('Feedback weights')
X = np.arange(0, w_feedback.shape[0])
Y = np.arange(0, w_feedback.shape[1])
X, Y = np.meshgrid(X, Y)
ax2.set_xlabel('Neuron number')
ax2.set_ylabel('Feedback number')
ax2.set_zlabel('W')
ax2.scatter(X, Y, w_feedback)

ax3 = fig.add_subplot(222, projection='3d')
ax3.set_title('Input signal')
X = np.arange(0, x.shape[0])
Y = np.arange(0, x.shape[1])
X, Y = np.meshgrid(X, Y)
ax3.set_xlabel('Neuron number')
ax3.set_ylabel('Time')
ax3.set_zlabel('Pulse')
ax3.plot_surface(X, Y, x.T, antialiased=False)

ax4 = fig.add_subplot(224, projection='3d')
ax4.set_title('Output signal')
X = np.arange(0, y.shape[0])
Y = np.arange(0, y.shape[1])
X, Y = np.meshgrid(X, Y)
ax4.set_xlabel('Neuron number')
ax4.set_ylabel('Time')
ax4.set_zlabel('Pulse')
ax4.plot_surface(X, Y, y.T, antialiased=False)

plt.show()
