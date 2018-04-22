import tensorflow as tf

x=tf.Variable([1,2])
a=tf.Variable([3,3])
#加法和减法OP
sub=tf.subtract(x,a)
add=tf.add(x,sub)
#初始化常量
init=tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(sub))
    print(sess.run(add))