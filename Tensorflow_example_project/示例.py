import tensorflow as tf
import numpy as np

#随机生成100个随机点
x_data=np.random.rand(100)
y_data=x_data*0.1+0.2

#构造线性模型
b=tf.Variable(1.)
k=tf.Variable(1.)
y=k*x_data+b

#二次代价函数  计算误差
# （y_data-y)的平方的平均值
loss=tf.reduce_mean(tf.square(y_data-y))

#定义个梯度下降法来进行训练的优化器    0.2是学习率
optimaizer=tf.train.GradientDescentOptimizer(0.2)

#最小化代价函数   optimaizer是上边定义的优化器
train=optimaizer.minimize(loss)

#初始化变量
init=tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for step in range(201):
        sess.run(train)  #开始减小误差
        if step%20==0:
            print(step,sess.run([k,b]))  #多个run要用列表

