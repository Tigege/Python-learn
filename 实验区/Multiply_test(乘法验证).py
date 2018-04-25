import tensorflow as tf
import numpy as np

x=tf.Variable([[1,1,1],[2,2,2]])
w=tf.Variable([[3,3],[4,4],[5,5]])
b=tf.Variable((tf.zeros([2,6])+0.1)*0.9)
c=tf.Variable(tf.zeros([2,6]))
tenst=tf.matmul(x, w)
init=tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print("the x is :\n"+str(sess.run(x)))
    print("the w is :\n"+str(sess.run(w)))
    print("fially is :\n"+str(sess.run(tenst)))



