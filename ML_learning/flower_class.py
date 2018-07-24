from skimage import io,transform
import glob
import os
import numpy as np
import tensorflow as tf
path="F:/datas/flower_photos/"
model_path="model.ckpt"

w=100
h=100
c=3

def read_img(path):
    cate=[path+x for x in os.listdir(path) if os.path.isdir(path+x)]
    imgs=[]
    labels=[]
    for idx,folder in enumerate(cate):
        for im in glob.glob(folder+'/*.jpg'):
            print("reading images:%s"%(im))
            img=io.imread(im)
            img=transform.resize(img,(w,h))
            imgs.append(img)
            labels.append(idx)
    return np.asarray(imgs,np.float32),np.asarray(labels,np.int32)

def model(input_tensor,train,regularizer):
    with tf.Variable_scope("layer1-conv1"):
        conv1_weights=tf.get_variable("weight",[5,5,3,32],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases=tf.get_variable("bias",[32],initializer=tf.constant_initializer(0.0))
        conv1=tf.nn.conv2d(input_tensor,conv1_weights,strides=[1,1,1,1],padding="SAME")
        relu1=tf.nn.relu(tf.nn.bias_add(conv1,conv1_biases))
    with tf.name_scope("layer1-pool1"):
        pool1=tf.nn.max_pool(relu1,ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID")
    with tf.Variable_scope("layer3-conv2"):
        conv2_weights=tf.get_variable("weight",[5,5,32,64],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("bias", [64], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding="SAME")
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
    with tf.name_scope("layer4-pool2"):
        pool2=tf.nn.max_pool(relu2,ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID")
    # s=np.int(num_example*ratio)
    with tf.Variable_scope("layer5-conv3"):
        conv3_weights = tf.get_variable("weight", [3, 3, 64, 128],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv3_biases = tf.get_variable("bias", [128], initializer=tf.constant_initializer(0.0))
        conv3 = tf.nn.conv2d(pool2, conv3_weights, strides=[1, 1, 1, 1], padding="SAME")
        relu3 = tf.nn.relu(tf.nn.bias_add(conv3, conv3_biases))
    with tf.name_scope("layer6-pool3"):
        pool3 = tf.nn.max_pool(relu3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

    with tf.Variable_scope("layer7-conv4"):
        conv4_weights = tf.get_variable("weight", [3, 3, 128, 128],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv4_biases = tf.get_variable("bias", [128], initializer=tf.constant_initializer(0.0))
        conv4 = tf.nn.conv2d(pool3, conv4_weights, strides=[1, 1, 1, 1], padding="SAME")
        relu4 = tf.nn.relu(tf.nn.bias_add(conv4, conv4_biases))
    with tf.name_scope("layer8-pool4"):
        pool4 = tf.nn.max_pool(relu4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
        nodes=6*6*128
        reshaped=tf.reshape(pool4,[-1,nodes])
        print("shape of reshaped:",reshaped.shape)


    with tf.Variable_scope("layer9-fc1"):
        fc1_weights=tf.get_variable("weight",[nodes,1024],initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer!=None:
        tf.add_to_collection("losses",regularizer(fc1_weights))
        fc1_biases=tf.get_variable("baise",[1024],initializer=tf.constant_initializer(0.1))
        fc1=tf.nn.relu(tf.matmul(reshaped,fc1_weights)+fc1_biases)
        if train:
            fc1=tf.nn.dropout(fc1,0.5)

    with tf.Variable_scope("layer10-fc2"):
        fc2_weights=tf.get_variable("weight",[1024,512],initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer!=None:
        tf.add_to_collection("losses",regularizer(fc2_weights))
        fc2_biases=tf.get_variable("baise",[512],initializer=tf.constant_initializer(0.1))
        fc2=tf.nn.relu(tf.matmul(reshaped,fc1_weights)+fc2_biases)
        if train:
            fc2=tf.nn.dropout(fc2,0.5)


    with tf.Variable_scope("layer11-fc3"):
        fc3_weights=tf.get_variable("weight",[512,5],initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer!=None:
        tf.add_to_collection("losses",regularizer(fc3_weights))
        fc3_biases=tf.get_variable("baise",[5],initializer=tf.constant_initializer(0.1))
        logit=tf.matmul(fc2,fc3_weights)+fc3_biases
    return logit


