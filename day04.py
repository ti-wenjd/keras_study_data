import keras
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

trX = np.linspace(-1, 1, 101)
trY = 3 * trX + np.random.randn(*trX.shape) * 0.33

#创建模型
model = Sequential()
model.add(Dense(input_dim=1, output_dim=1, init='uniform', activation='linear'))

#初始化的权重值
weights = model.layers[0].get_weights()
w_init = weights[0][0][0]
b_init = weights[1][0]
print('Linear regression model is initialized with weights w: %.2f, b: %.2f' % (w_init, b_init))

#优化器和损失函数
model.compile(optimizer='sgd', loss='mse')


#训练模型
model.fit(trX, trY, nb_epoch=200, verbose=1)


#训练后的权重值
weights = model.layers[0].get_weights()
w_final = weights[0][0][0]
b_final = weights[1][0]
print('Linear regression model is trained to have weight w: %.2f, b: %.2f' % (w_final, b_final))