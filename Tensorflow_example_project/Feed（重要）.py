import tensorflow as tf

#创建占位符  并定义数据格式
input1=tf.placeholder(tf.float32)
input2=tf.placeholder(tf.float32)

output=tf.multiply(input1,input2)

with tf.Session() as sess:
    #feed_dict 字典形式导入 8.  代表8.0
    print(sess.run(output,feed_dict={input1:[8.],input2:[2.]}))