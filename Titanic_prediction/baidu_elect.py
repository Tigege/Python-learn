# coding: utf-8
import tensorflow as tf
import numpy as np
import baidu_test

# 载入数据集

# 每个批次的大小
batch_size = 100
# 计算一共有多少个批次

new_data=baidu_test.make_data()
new_data = new_data.values.astype(np.float32)       # change to numpy array and float32
np.random.shuffle(new_data)
sep = int(0.7*len(new_data))
train_data = new_data[:sep]                         # training data (70%)
test_data = new_data[sep:]                          # test data (30%)

tf_input = tf.placeholder(tf.float32, [None, 8], "input")
tfx = tf_input[:, :6]
tfy = tf_input[:, 6:]

keep_prob = tf.placeholder(tf.float32)
lr = tf.Variable(0.001, dtype=tf.float32)

# 创建一个简单的神经网络
W1 = tf.Variable(tf.truncated_normal([6, 500], stddev=0.1))
b1 = tf.Variable(tf.zeros([500]) + 0.1)
L1 = tf.nn.tanh(tf.matmul(tfx, W1) + b1)
L1_drop = tf.nn.dropout(L1, keep_prob)

W2 = tf.Variable(tf.truncated_normal([500, 300], stddev=0.1))
b2 = tf.Variable(tf.zeros([300]) + 0.1)
L2 = tf.nn.tanh(tf.matmul(L1_drop, W2) + b2)
L2_drop = tf.nn.dropout(L2, keep_prob)

W3 = tf.Variable(tf.truncated_normal([300, 2], stddev=0.1))
b3 = tf.Variable(tf.zeros([2]) + 0.1)
prediction = tf.nn.softmax(tf.matmul(L2_drop, W3) + b3)

# 交叉熵代价函数
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tfy, logits=prediction))
# 训练
train_step = tf.train.AdamOptimizer(lr).minimize(loss)

# 初始化变量
init = tf.global_variables_initializer()

# 结果存放在一个布尔型列表中
correct_prediction = tf.equal(tf.argmax(tfy, 1), tf.argmax(prediction, 1))  # argmax返回一维张量中最大的值所在的位置
# 求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(51):
        sess.run(tf.assign(lr, 0.001 * (0.95 ** epoch)))
        learning_rate = sess.run(lr)
        for batch in range(1000):
            sess.run(train_step, feed_dict={tf_input:train_data, keep_prob: 1.0})
            acc = sess.run(accuracy, feed_dict={tf_input: test_data, keep_prob: 1.0})
            print("Iter " + str(epoch) + ", Testing Accuracy= " + str(acc) + ", Learning Rate= " + str(learning_rate))







