import tensorflow as tf
import numpy as np
import baidu_test

new_data=baidu_test.make_data()

new_data = new_data.values.astype(np.float32)
np.random.shuffle(new_data)

sep = int(0.7*len(new_data))
train_data = new_data[:sep]                         # training data (70%)
test_data = new_data[sep:]                          # test data (30%)

tf_input = tf.placeholder(tf.float32, [None, 8], "input")
tfx = tf_input[:, :6]
tfy = tf_input[:, 6:]

l1 = tf.layers.dense(tfx, 500, tf.nn.relu, name="l1")
l2 = tf.layers.dense(l1, 900, tf.nn.relu, name="l2")
l3 = tf.layers.dense(l2, 800, tf.nn.relu, name="l3")
l4 = tf.layers.dense(l3, 500, tf.nn.relu, name="l4")
out = tf.layers.dense(l4, 2, name="out")
prediction = tf.nn.softmax(out, name="pred")
init = tf.global_variables_initializer()
loss = tf.losses.softmax_cross_entropy(onehot_labels=tfy, logits=out)
accuracy = tf.metrics.accuracy(          # return (acc, update_op), and create 2 local variables
    labels=tf.argmax(tfy, axis=1), predictions=tf.argmax(out, axis=1),)[1]
opt = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train_op = opt.minimize(loss)

with tf.Session() as sess:
    sess.run(init)
    for step in range(5000):
        sess.run(train_op,feed_dict={tf_input:train_data})
        print(sess.run(accuracy, feed_dict={tf_input: test_data}))
        # if step%50==0:



