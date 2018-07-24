import tensorflow as tf
import numpy as np

x_data=np.random.rand(100).astype(np.float32)
y_data=x_data*0.2+0.3
print(x_data)
weights=tf.Variable(tf.random_uniform([1],-1.0,1.0))
# print("weights",tf.random_uniform([1],-1.0,1.0))
baises=tf.Variable(tf.zeros([1]))
y=x_data*weights+baises
loss=tf.reduce_mean(tf.square(y-y_data))
testdemo=tf.square(y-y_data)
train=tf.train.GradientDescentOptimizer(0.1).minimize(loss)
init=tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print("testdemo")
    print(sess.run(testdemo))
    print("losse")
    print(sess.run(loss))
    print(sess.run([weights,baises]))
    # for i in range(500):
    #     sess.run(train)
    #     if i %10==0:
    #         print(sess.run(loss))
    #         print(sess.run(weights),sess.run(baises))


