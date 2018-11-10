import numpy as np

np.random.seed(1337)  # for reproducibility
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop

# download the mnist to the path '~/.keras/datasets/' if it is the first time to be called
# X shape (60,000 28x28), y shape (10,000, )
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print(X_train.shape[0])
# data pre-processing
X_train = X_train.reshape(X_train.shape[0], -1) / 255.  # normalize
X_test = X_test.reshape(X_test.shape[0], -1) / 255.  # normalize
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

print(X_train.shape)

print(y_train[:9])

# Another way to build your neural net 建立神经网络
# 第一段就是加入 Dense 神经层。32 是输出的维度，784 是输入的维度。
#  第一层传出的数据有 32 个 feature，传给激励单元，激励函数用到的是 relu 函数。
#  经过激励函数之后，就变成了非线性的数据。 然后再把这个数据传给下一个神经层，
#  这个 Dense 我们定义它有 10 个输出的 feature。同样的，此处不需要再定义输入的维度
# ，因为它接收的是上一层的输出。 接下来再输入给下面的 softmax 函数，用来分类。
model = Sequential([
    Dense(32, input_dim=784),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
])

# Another way to define your optimizer 优化器
rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)


#激活模型
#优化器，可以是默认的，也可以是我们在上一步定义的。
# 损失函数，分类和回归问题的不一样，用的是交叉熵。
#  metrics，里面可以放入需要计算的 cost，accuracy，score 等
model.compile(optimizer=rmsprop,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print('Training ------------')
# Another way to train the model
model.fit(X_train, y_train, epochs=2, batch_size=32)

print('\nTesting ------------')
# Evaluate the model with the metrics we defined earlier
loss, accuracy = model.evaluate(X_test, y_test)

print('test loss: ', loss)
print('test accuracy: ', accuracy)
