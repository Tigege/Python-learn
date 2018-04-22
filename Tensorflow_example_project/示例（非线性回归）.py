import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#随机生层200个随机点  在 -0.5到0.5之间 200个
x_data=np.linspace(-0.5,0.5,200)[:,np.newaxis]
noise=np.random.normal(0,0.02,x_data.shape)
y_data=np.square(x_data)+noise

#[None,1]  行数不确定，一列
x=tf.placeholder(tf.float32,[None,1])
y=tf.placeholder(tf.float32,[None,1])

#定义中间层
#一行10列
Weights_l1=tf.Variable(tf.random_normal([1,10]))
biases_l1=tf.Variable(tf.zeros([1,10]))
Wx_plus_b_l1=tf.matmul(x,Weights_l1)+biases_l1
L1=tf.nn.tanh(Wx_plus_b_l1)  #激励函数

#定义输出层
Weights_l2=tf.Variable(tf.random_normal([10,1]))
biases_l2=tf.Variable(tf.zeros([1,1]))
Wx_plus_b_l2=tf.matmul(L1,Weights_l2)+biases_l2
prediction=tf.nn.tanh(Wx_plus_b_l2)  #得到预测值  激励函数

#二次代价函数
loss=tf.reduce_mean(tf.square(y-prediction))
#使用梯度下降优化器训练
train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(2000):
        sess.run(train_step,feed_dict={x:x_data,y:y_data})

    #获得预测值
    prediction_value=sess.run(prediction,feed_dict={x:x_data})
    plt.figure()
    plt.scatter(x_data,y_data)
    #'r-' 红色实线   lw=5 宽度为5
    plt.plot(x_data,prediction_value,'r-',lw=5)
    plt.show()


