import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import baidu_test
# read data from file
data = baidu_test.make_data()
# select features and labels for training
dataset_X = data[["K1K2驱动信号","电子锁驱动信号","急停信号","门禁信号","THDV-M","THDI-M","test1","test2","test3","test4"]].as_matrix()
dataset_Y = data[["label","label2"]].as_matrix()
print("--------------")
print(dataset_X)
# split training data and validation set data
X_train, X_val, y_train, y_val = train_test_split(dataset_X, dataset_Y,
                                                  test_size=0.2,
                                                  random_state=42)

# create symbolic variables
X = tf.placeholder(tf.float32, shape=[None, 10])
y = tf.placeholder(tf.float32, shape=[None, 2])


l1 = tf.layers.dense(X, 1000, tf.nn.tanh, name="l1")
l2 = tf.layers.dense(l1, 900, tf.nn.tanh, name="l2")
l3 = tf.layers.dense(l2, 400, tf.nn.tanh, name="l3")
l4 = tf.layers.dense(l3, 100, tf.nn.tanh, name="l4")

out = tf.layers.dense(l4, 2, name="out")
y_pred = tf.nn.softmax(out, name="pred")

# weights and bias are the variables to be trained
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_pred))
# 训练
train_step = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)
# 初始化变量
init = tf.global_variables_initializer()

# 结果存放在一个布尔型列表中
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))  # argmax返回一维张量中最大的值所在的位置
# 求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# use session to run the calculation
with tf.Session() as sess:
    sess.run(init)
    step=0
    while True:
        step=step+1
        sess.run(train_step, feed_dict={X: X_train, y: y_train})
        if step % 100 == 0:
            acc=sess.run(accuracy,feed_dict={X: X_val, y: y_val})
            print("acc is :"+str(acc))
            if acc >0.95:
                break


    subdata,Id=baidu_test.read_data()
    #print(Id.as_matrix())
    prediction = np.argmax(sess.run(y_pred, feed_dict={X: subdata}),1)
    for i in range(len(prediction)):
        if prediction[i]==0:
            prediction[i]=1
        else:
            prediction[i]=0
    submission = pd.DataFrame({
        "ID": Id,
        "predictrion": prediction
    })
    submission.to_csv("baidu_sub2.csv",index=False)
    print("over")