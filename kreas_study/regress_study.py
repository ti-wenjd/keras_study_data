import numpy as np

np.random.seed(1337)  # for reproducibility
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt  # 可视化模块

# create some data
X = np.linspace(-1, 1, 200)
np.random.shuffle(X)  # randomize the  类似洗牌  重新洗数据 搅乱顺序
Y = 0.5 * X + 2 + np.random.normal(0, 0.05, (200,))
# plot data


X_train, Y_train = X[:160], Y[:160]  # train 前 160 data points   训练数据
X_test, Y_test = X[160:], Y[160:]  # test 后 40 data points       测试数据

model = Sequential()
model.add(Dense(output_dim=1, input_dim=1))

# choose loss function and optimizing method 误差函数用的是 mse 均方误差；优化器用的是 sgd 随机梯度下降法
model.compile(loss='mse', optimizer='sgd')

print('Training -----------')
for step in range(301):
    cost = model.train_on_batch(X_train, Y_train)

    if step % 100 == 0:
        print('train cost: ', cost)

Y_pred = model.predict(X_test)
plt.scatter(X_test, Y_test)
plt.plot(X_test, Y_pred)
plt.show()
