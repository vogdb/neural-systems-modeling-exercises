import matplotlib.pyplot as plt

pls = [0, 0, 1, 0, 0]
x_gill = pls * 6
x_tail = [0] * len(x_gill)
# only one pulse to the tail in the middle of simulation
x_tail[len(x_tail) // 2] = 1
y = [0] * len(x_gill)

w0 = 4
w = w0
w_habit = 0.7

time = range(0, len(x_gill))
for t in time:
    y[t] = x_gill[t] * w
    if x_gill[t] > 0:
        w = w * w_habit
    if x_tail[t] > 0:
        w = w0

f, subplots = plt.subplots(2, 1, sharex='all')

subplots[0].set_xlabel('t')
subplots[0].set_ylabel('x')
subplots[0].plot(time, x_gill)

subplots[1].set_xlabel('t')
subplots[1].set_ylabel('y')
subplots[1].plot(time, y)

plt.show()
