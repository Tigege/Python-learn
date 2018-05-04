import tensorflow as tf
import numpy as np
import pandas as pd

#数据预处理
def read_data():
    data=pd.read_csv('train.csv')  #pandas 读取
    data=data.fillna(0)     #NAN   填入0

    datax=data[['Sex', 'Age', 'Pclass', 'SibSp', 'Parch', 'Fare','Embarked']]  #pandas选择列
    datax=pd.get_dummies(datax)            # one—hot 编码
    data['Deceased'] = data['Survived'].apply(lambda s: 1 - s)  #one -hot编码
    datay = data[['Survived','Deceased']]
    return datax,datay


if __name__=="__main__":
    new_datax,new_datay=read_data()
    new_dataxx = new_datax.values.astype(np.float32)  #把pandas矩阵  转化为np矩阵没有二维标签变为存矩阵
    new_datayy = new_datay.values.astype(np.float32)  # 把pandas矩阵  转化为np矩阵没有二维标签变为存矩阵
    np.random.shuffle(new_dataxx)   #随机打乱顺序
    np.random.shuffle(new_datayy)  # 随机打乱顺序

    sep = int(0.7 * len(new_dataxx))   #拆分数据集
    train_datax = new_dataxx[:sep]  # training data (70%)
    train_datay=new_datayy[:sep]
    test_datax = new_dataxx[sep:]  # test data (30%)
    test_datay=new_datayy[sep:]

    # new_datax.to_csv("Taitan_onehot.csv", index=False)
    #简便方法   打乱+拆分
    '''
    分割的简单方法    from sklearn.model_selection import train_test_split  
    X_train, X_val, y_train, y_val = train_test_split(dataset_X, dataset_Y,
                                                  test_size=0.2,
                                                  random_state=42)
    '''

    #全链接神经层 的输入   tf_inputx 训练数据的特征信息     tf_inputy训练数据的标签
    tf_inputx=tf.placeholder(tf.float32, [None, 11])
    tf_inputy=tf.placeholder(tf.float32,[None,2])

    #搭建全链接神经网络
    l1 = tf.layers.dense(tf_inputx, 600, tf.nn.tanh, name="l1")
    l2 = tf.layers.dense(l1, 400, tf.nn.tanh, name="l2")
    out = tf.layers.dense(l2, 2, name="l3")

    #计算概率    相加为1
    prediction=tf.nn.softmax(out, name="pred")

    #计算误差  代价函数
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_inputy, logits=prediction))
    #优化器  减少误差
    train_step = tf.train.GradientDescentOptimizer(0.6).minimize(loss)
    #初始化变量
    init = tf.global_variables_initializer()

    # 结果存放在一个布尔型列表中
    correct_prediction = tf.equal(tf.argmax(test_datay, 1), tf.argmax(prediction, 1))  # argmax返回一维张量中最大的值所在的位置
    # 求准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    '''
    correct_prediction = tf.equal(tf.argmax(tf_inputy, 1), tf.argmax(prediction, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    '''
    #创建会话
    with tf.Session() as sess:
        sess.run(init)  #初始化变量
        for i in range(5000):   #控制迭代次数

            # 启动优化其  减少误差 开始训练
            sess.run(train_step, feed_dict={tf_inputx: train_datax, tf_inputy: train_datay})

            # 查看当前训练实时数据反馈
          #  print("number:"+str(i))


            acc = sess.run(accuracy, feed_dict={tf_inputx: test_datax, tf_inputy: test_datay})
            print("Accuracy on validation set:" +str(acc))
           # acc = sess.run(accuracy, feed_dict={tf_inputx: test_datax, tf_inputy: test_datay})
           # print("Iter" + str(i) + ",Testing Accuracy" + str(acc))
        #训练完成后  进行预测 查看预测结果
        print("test")
        print(test_datay)
        print(sess.run(prediction,feed_dict={tf_inputx:test_datax}))



