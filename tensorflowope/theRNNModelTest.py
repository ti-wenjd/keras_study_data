import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(1)

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

lr = 0.001
training_iters = 100000
batch_size = 128

n_inputs = 28  # 输入的数据shape   MNIST data input (img shape: 28*28)  代表图片宽
n_steps = 28  # 时间步长    代表图片高
n_hidden_units = 128  # 隐藏层
n_classes = 10  # MNIST classes (0-9 digits)  输出的数据格式

# 占位符
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])  # None表示第一维是任意数量  这是个3维的数据
y = tf.placeholder(tf.float32, [None, n_classes])

# 权重系数
weights = {
    # (28,128)   random_normal取的是正态分布的随机数值  默认值是1以内
    "in": tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    # (128,10)
    "out": tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}

# 移动的常量
biases = {
    # (128,)
    "in": tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    # (10,)
    "out": tf.Variable(tf.constant(0.1, shape=[n_classes, ]))

}


def RNN(X, weights, biases):
    # 原始的 X 是 3 维数据, 我们需要把它变成 2 维数据才能使用 weights 的矩阵乘法
    # X ==> (128 batch * 28 steps, 28 inputs)
    # 转换数据的shape 将输入的数据shape 转换成设定的维度  -1 表示第一个维度自动补全  ?*28 的数据
    X = tf.reshape(X, [-1, n_inputs])

    # 隐藏层
    # X_in = W * X + b
    X_in = tf.matmul(X, weights["in"] + biases["in"])

    # X_in ==> (128 batches, 28 steps, 128 hidden) 换回3维
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])

    # cell
    # 使用 basic LSTM Cell.
    cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units)

    # 初始化全零 state
    init_state = cell.zero_state(batch_size, dtype=tf.float32)

    outputs, final_state = tf.nn.dynamic_rnn(cell, X_in, initial_state=init_state, time_major=False)

    outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))

    results = tf.matmul(outputs[-1], weights["out"]) + biases["out"]

    return results


# 其中 x是个三维的数据输入
pred = RNN(x, weights, biases)

# 定义损失函数
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

# 生成训练的步长
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

# 正确的预测
# 启动argmax是返回当前最大值的下标索引  后面的 1对应的是axis的值 也就是轴  0是竖着计算  1是横着计算
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

with tf.Session() as sess:
    # 初始化
    init = tf.global_variables_initializer()
    sess.run(init)

    #
    step = 0
    while step * batch_size < training_iters:
        # 生成数据
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        # 江取出的数据x 变成3维的
        batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
        # 喂数据  x y 分别是上边定义的占位符 其中x是三维的数据  y是二维的
        sess.run([train_op], feed_dict={x: batch_xs, y: batch_ys})

        if step % 20 == 0:
            print(sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys}))

        step += 1
