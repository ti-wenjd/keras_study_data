import tensorflow as tf
import numpy as np
"""
https://www.cnblogs.com/cvtoEyes/p/9003783.html
"""
# [2,3]是数据的shape 表示第一个维度2个  第二个维度3个
w = tf.Variable(tf.random_uniform([2, 3], minval=0,maxval=10,seed=1))
x = tf.Variable(tf.random_normal([2, 3]))
y = tf.Variable(tf.random_shuffle(np.arange(6).reshape((2,3))))


init = tf.global_variables_initializer()

sess = tf.Session()

sess.run(init)

print(sess.run(y))  # 指定形状的张量填充随机正常值
