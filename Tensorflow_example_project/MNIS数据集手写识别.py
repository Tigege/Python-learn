import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.truncated_normal([in_size, out_size],stddev=0.1))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.nn.softmax(tf.matmul(inputs, Weights) + biases)
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

mnist=input_data.read_data_sets("MNIST_data",one_hot=True)
batch_size=100
#计算一共有多少个批次
n_batch=mnist.train.num_examples//batch_size
lr=tf.Variable(0.001,dtype=tf.float32)
x=tf.placeholder(tf.float32,[None,784])
y=tf.placeholder(tf.float32,[None,10])
l1=add_layer(x,784,500)
l2=add_layer(l1,500,300)
prediction=add_layer(l2,300,10,activation_function=None)

#二次迭代
#loss=tf.reduce_mean(tf.square(y-prediction))

#对数释然函数
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
#梯度下降
train_step = tf.train.AdamOptimizer(lr).minimize(loss)

#初始化
init = tf.global_variables_initializer()

#结果存放在一个布尔列表中
correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))

#求准确率
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(21000):
        sess.run(tf.assign(lr, 0.001 * (0.95 ** epoch)))
        for batch in range(n_batch):

            batch_xs,batch_ys=mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys})
        acc=sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        print("Iter"+str(epoch)+",Testing Accuracy"+str(acc))

