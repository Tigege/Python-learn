import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
#从CSV文件读取
data=pd.read_csv('train.csv')
# print(data.info())
# print(data)

data['Sex']=data['Sex'].apply(lambda s: 1 if s =='male' else 0)
data=data.fillna(0)
#print(data)
dataset_X=data[['Sex','Age','Pclass','SibSp','Parch','Fare']]
dataset_X=dataset_X.as_matrix()
# print('sdf')
# print(dataset_X)
data['Deceased']=data['Survived'].apply(lambda  s: int(not s))
dataset_Y=data[['Deceased','Survived']]
# print(dataset_Y)
#as_matrix()  转化为纯矩阵
dataset_Y=dataset_Y.as_matrix()
# print("aasdf")
# print(dataset_X)

#切分数据集  乱序拆分 验证数据占20%
X_train, X_test, y_train, y_test = train_test_split(dataset_X, dataset_Y,
                                                 test_size=0.2,
                                                  random_state=42)

#输入占位符   用于训练数据的输入
X=tf.placeholder(tf.float32,shape=[q,6])  #一组数据6个输入
Y=tf.placeholder(tf.float32,shape=[None,2])  #

W=tf.Variable(tf.random_normal([6,2]),name='weights')
b=tf.Variable(tf.zeros([2]),name='bias')

y_pred=tf.nn.softmax(tf.matmul(input,W)+b)


cross_entropy = - tf. reduce_sum(Y * tf.log(y_pred + 1e-10),
                                reduction_indices=1)
cost = tf.reduce_mean(cross_entropy)

train_op=tf.train.GradientDescentOptimizer(0.001).minimize(cost)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for epoch in range(10):
        total_loss=0.



