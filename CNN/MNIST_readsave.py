import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets('MNIST_data',one_hot=True)
batch_size=100
n_batch=mnist.train.num_examples//batch_size

def weight_variable(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding="SAME")
    #输入x [batch,in_height,in_with,in_channels]  batch批次 1000  height 图片的长 width宽 channels通道数
    #channels RGB则为3  黑白为1

    #输入W [height,width,in_channels,out_channels]滤波器 卷积核 长、宽、输入通道数、输出通道数

    #strides 步长 位置[0][3]为1  [1]代表X横向步长  [2]代表Y方向的步长
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
    #ksize窗口的大小 [1,X,Y,1]  X、Y代表池化窗口大小

x=tf.placeholder(tf.float32,[None,784])
y=tf.placeholder(tf.float32,[None,10])

#把一列数值转化为2维的形式 来进行池化 [batch,in_height,in_width,in_channels]
x_image=tf.reshape(x,[-1,28,28,1])

W_conv1=weight_variable([5,5,1,32])#5x5的采样窗口 32个卷积核 从1个平面抽取特征
b_conv1=bias_variable([32])  #每一个卷积核的偏置值

h_conv1=tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
h_pool1=max_pool_2x2(h_conv1)

W_conv2=weight_variable([5,5,32,64])#5x5的采样窗口 64个卷积核 从32个平面抽取特征
b_conv2=bias_variable([64])  #每一个卷积核的偏置值

h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
h_pool2=max_pool_2x2(h_conv2)

#全连接层
W_fc1=weight_variable([7*7*64,1024])
b_fc1=bias_variable([1024])

h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64])
h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)

keep_prob=tf.placeholder(tf.float32)
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)

W_fc2=weight_variable([1024,10])
b_fc2=bias_variable([10])

prediction=tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)

cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction=tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))

accracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
saver=tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(accracy,feed_dict={x:mnist.test.images,y:mnist.test.labels,keep_prob:1.0}))
    saver.restore(sess,"Model/MNIST_net.ckpt")
    print(sess.run(accracy, feed_dict={x: mnist.test.images, y: mnist.test.labels,keep_prob:1.0}))