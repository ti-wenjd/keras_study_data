import tensorflow as tf
import numpy as np

# 读取
W = tf.Variable(np.arange(6).reshape((2, 3)), dtype=tf.float32, name="weights")
b = tf.Variable(np.arange(3).reshape((1, 3)), dtype=tf.float32, name="biases")

# 这里不需要初始化步骤 init= tf.initialize_all_variables()

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, "my_net/save_net.ckpt")
    print("weight:", sess.run(W))
    print("biases:", sess.run(b))
