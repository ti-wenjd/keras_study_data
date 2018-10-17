##一些基本概念
import numpy as np
import keras
from keras import backend as K
from keras import models
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import LSTM, Embedding
from keras.layers import Dense, Dropout, Activation

import tensorflow as tf
import random as rn
import os


# 沿着某个轴 的功能 （对张量的理解）
def axisFun():
    """
    规模最小的张量是0阶张量，即标量，也就是一个数。
    当我们把一些数有序的排列起来，就形成了1阶张量，也就是一个向量
    如果我们继续把一组向量有序的排列起来，就形成了2阶张量，也就是一个矩阵
    把矩阵摞起来，就是3阶张量，我们可以称为一个立方体，具有3个颜色通道的彩色图片就是一个这样的立方体
    把立方体摞起来，好吧这次我们真的没有给它起别名了，就叫4阶张量了，不要去试图想像4阶张量是什么样子，它就是个数学上的概念。

    张量的阶数有时候也称为维度，或者轴，轴这个词翻译自英文axis。
    譬如一个矩阵[[1,2],[3,4]]，是一个2阶张量，有两个维度或轴，
    沿着第0个轴（为了与python的计数方式一致，本文档维度和轴从0算起）你看到的是[1,2]，[3,4]两个向量，沿着第1个轴你看到的是[1,3]，[2,4]两个向量。
    :return:
    """
    a = np.array([[1, 2], [3, 4]])
    sum0 = np.sum(a, axis=0)
    sum1 = np.sum(a, axis=1)

    print(sum0)
    print(sum1)


# 如何从Sequential模型中去除一个层？
def remove_Dense_from_model():
    model = Sequential()
    model.add(Dense(32, activation='relu', input_dim=784))
    model.add(Dense(32, activation='relu'))

    print(model.layers)  # "2"

    model.pop()
    print(model.layers)  # "1"


def keshihua():
    os.environ['PYTHONHASHSEED'] = '0'
    # The below is necessary for starting Numpy generated random numbers
    # in a well-defined initial state.

    np.random.seed(42)

    # The below is necessary for starting core Python generated random numbers
    # in a well-defined state.
    rn.seed(12345)
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

    tf.set_random_seed(1234)

    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)

    print("ok")


def quickStart():
    model = Sequential()
    model.add(Dense(32, activation='relu', input_dim=100))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Generate dummy data
    import numpy as np
    data = np.random.random((1000, 100))
    labels = np.random.randint(2, size=(1000, 1))

    # Train the model, iterating on the data in batches of 32 samples
    model.fit(data, labels, epochs=10, batch_size=32)


# 基于多层感知器的softmax多分类：
def duo_ceng_gan_zhi():
    x_train = np.random.random((1000, 20))
    y_train = keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)
    x_test = np.random.random((100, 20))
    y_test = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)

    model = Sequential()
    # Dense(64) is a fully-connected layer with 64 hidden units.
    # in the first layer, you must specify the expected input data shape:
    # here, 20-dimensional vectors.
    model.add(Dense(64, activation='relu', input_dim=20))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              epochs=20,
              batch_size=128)
    score = model.evaluate(x_test, y_test, batch_size=128)


# MLP的二分类：
def MLP_ER_FEN_LEI():
    x_train = np.random.random((1000, 20))
    y_train = np.random.randint(2, size=(1000, 1))
    x_test = np.random.random((100, 20))
    y_test = np.random.randint(2, size=(100, 1))

    model = Sequential()
    model.add(Dense(64, input_dim=20, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    model.fit(x_train, y_train,
              epochs=20,
              batch_size=128)
    score = model.evaluate(x_test, y_test, batch_size=128)


def lstm_caipiao_pre():
    data_dim = 16
    timesteps = 8
    num_classes = 10

    # expected input data shape: (batch_size, timesteps, data_dim)
    model = Sequential()
    model.add(LSTM(32, return_sequences=True,
                   input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
    model.add(LSTM(32, return_sequences=True))  # returns a sequence of vectors of dimension 32
    model.add(LSTM(32))  # return a single vector of dimension 32
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    # Generate dummy training data
    x_train = np.random.random((1000, timesteps, data_dim))
    y_train = np.random.random((1000, num_classes))

    # Generate dummy validation data
    x_val = np.random.random((100, timesteps, data_dim))
    y_val = np.random.random((100, num_classes))

    model.fit(x_train, y_train,
              batch_size=64, epochs=5,
              validation_data=(x_val, y_val))


if __name__ == '__main__':
    x_train = np.random.random((50, 8, 16))
    print(x_train)
