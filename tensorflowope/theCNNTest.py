import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)


##定义权重
def weight_var(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# 定义biases
def bias_var(shape):
    intial = tf.constant(0.1, shape=shape)
    return tf.Variable(intial)


# 定义二维矩阵
def conv2d(x, W):
    # x W 分别为x轴 和 Y轴 移动的步长
    # strides [1,x_movement,y_movement,1]
    # Must have strides[0]=strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1, ], padding="SAME")


# 池化
def max_pool_var(x):
    # x W 分别为x轴 和 Y轴 移动的步长
    # strides [1,x_movement,y_movement,1]
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


sess = tf.Session()


# 校验
def compute_accuacy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    # 1表示横轴
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result


def add_layer(inputs, in_size, out_size, layer_name, activation_function=None, ):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, )
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    # here to dropout  解决过拟合的问题  使用dropout
    Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b, )
    tf.summary.histogram(layer_name + '/outputs', outputs)
    return outputs


# 定义变量
xs = tf.placeholder(tf.float32, [None, 784])/255
ys = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

x_image = tf.reshape(xs, [-1, 28, 28, 1])

## conv1 layer ## 第一层卷积
W_conv1 = weight_var([5,5, 1,32]) # patch 5x5, in size 1, out size 32
b_conv1 = bias_var([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # output size 28x28x32
h_pool1 = max_pool_var(h_conv1)



## conv2 layer ## 第二层卷积
W_conv2 = weight_var([5,5, 32, 64]) # patch 5x5, in size 32, out size 64
b_conv2 = bias_var([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # output size 14x14x64
h_pool2 = max_pool_var(h_conv2)

#建立全连接层
W_fc1 = weight_var([7*7*64, 1024])
b_fc1 = bias_var([1024])
# [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


## fc2 layer ##
W_fc2 = weight_var([1024, 10])
b_fc2 = bias_var([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


#定义损失函数  正确值和预测值之间的差
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))
#训练步长
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

#初始化
init = tf.global_variables_initializer()
sess.run(init)

for i in range(1000):
    batch_xs,batch_ys = mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys,keep_prob:0.5})
    if i % 50 == 0:
        print(compute_accuacy(mnist.test.images[:1000], mnist.test.labels[:1000]))