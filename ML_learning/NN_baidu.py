import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import comments_return_2
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
# read data from file
data,dataset_Y=comments_return_2.test_data()

# split training data and validation set data
X_train, X_val, y_train, y_val = train_test_split(data, dataset_Y,
                                                  test_size=0.4,
                                                  random_state=42)
feature_extraction=TfidfVectorizer()
X_train=feature_extraction.fit_transform(X_train)
y_train=feature_extraction.transform(y_train)
print(X_train.shape)
'''

# create symbolic variables  
X = tf.placeholder(tf.float32, shape=[None, 6])
y = tf.placeholder(tf.float32, shape=[None, 3])

l1 = tf.layers.dense(X, 50, tf.nn.tanh, name="l1")
l2 = tf.layers.dense(l1, 50, tf.nn.tanh, name="l2")
l3 = tf.layers.dense(l2, 20, tf.nn.tanh, name="l3")
out = tf.layers.dense(l3, 2, name="out")
y_pred = tf.nn.softmax(out, name="pred")
# weights and bias are the variables to be trained  
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_pred))
# 训练  
train_step = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)
# 初始化变量  
init = tf.global_variables_initializer()

# 结果存放在一个布尔型列表中  
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))  # argmax返回一维张量中最大的值所在的位置  
# 求准确率  
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# use session to run the calculation  
with tf.Session() as sess:
    sess.run(init)
    for step in range(1002):
        sess.run(train_step, feed_dict={X: X_train, y: y_train})
        if step % 100 == 0:
            acc = sess.run(accuracy, feed_dict={X: X_val, y: y_val})
            print("acc is :" + str(acc))
'''
