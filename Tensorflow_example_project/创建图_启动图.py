import tensorflow as tf

#创建一个常量OP
a1=tf.constant([[3,3]])
a2=tf.constant([[2],[3]])

#创建一个矩阵乘法OP，把a1和a2传入
product=tf.matmul(a1,a2)

with tf.Session() as sess:
    result = sess.run(product)
    print(result)
'''
#定义一个会话启动默认的图
sess=tf.Session()
#调用sess的run方法执行矩阵乘法OP
#run(product)触发了图中3个OP
result=sess.run(product)
print(result)
sess.close()
'''


