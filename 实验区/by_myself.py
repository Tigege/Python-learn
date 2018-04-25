import tensorflow as tf
import numpy as np
x_data =np.random.rand(100).astype(np.float32)
y_data=x_data*0.1 +0.3

Weights=tf.Variable(tf.random_uniform([1],-1.0,1.0))
biases=tf.Variable(tf.zeros([1]))


'''
y=np.linspace(-0.5,0.5,200)
x=np.linspace(-0.5,0.5,200)[:,np.newaxis]
noise=np.random.normal(0,0.2,x.shape)
'''
'''
y=x*2+noise

testx=tf.placeholder(tf.float32,[None,1])
testy=tf.placeholder(tf.float32,[None,1])
data=testx*
# print(noise)
# print(y)
# print(x)
'''