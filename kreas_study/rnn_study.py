import numpy as np

np.random.seed(1337)  # for reproducibility
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, TimeDistributed, Dense
from keras.optimizers import Adam

batch_start = 0
time_steps = 20
batch_size = 50
input_size = 1
out_size = 1
cell_size = 20
lr = 0.006


def get_batch():
    global batch_start, time_steps

    xs = np.arange(batch_start, batch_start + batch_size * time_steps).reshape((batch_size, time_steps)) / (10 * np.pi)

    seq = np.sin(xs)
    res = np.cos(xs)

    batch_start += time_steps

    return [seq[:, :, np.newaxis], res[:, :, np.newaxis], xs]


model = Sequential()

# 创建LSTM RNN
model.add(LSTM(batch_input_shape=(batch_size, time_steps, input_size), return_sequences=True, units=out_size,
               stateful=True))

model.add(TimeDistributed(Dense(out_size)))
adam = Adam(lr)
model.compile(optimizer=adam, loss="mse")

print("----------train----------")

# 训练
for step in range(501):
    x_batch, y_batch, xs = get_batch()
    cost = model.train_on_batch(x_batch, y_batch)
    pred = model.predict(x_batch, batch_size)

    plt.plot(xs[0, :], y_batch[0].flatten(), "r", xs[0, :], pred.flatten()[:time_steps], "b--")
    plt.ylim((-1.2, 1.2))
    plt.draw()
    plt.pause(0.1)
    if step % 10 == 0:
        print("train cost", cost)
