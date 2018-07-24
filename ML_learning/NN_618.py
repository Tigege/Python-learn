import numpy as np
import pandas as pd
import jieba
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
import sys
sys.setrecursionlimit(10000) #增大堆栈最大深度(递归深度)，据说默认为1000，报错

datas = pd.read_csv("./datas/data_train.csv", encoding='gbk', delimiter="\t",header=None)
all_ = datas.fillna("无")
all_["word"] = all_[2].apply(lambda s: list(jieba.cut(s)))  ###<span style="color:#ff0000;">语料分词</span>
print(all_)
maxlen = 100  # 截断字数
min_count = 5 #去除低频词，单句切成词数要比切成字数少的多，相应的要去掉的低频词频率也要比低频字低一些
content = [] #词表
for i in all_['word']:
    content.extend(i)
abc = pd.Series(list(content)).value_counts() #统计词频
abc = abc[abc >= min_count] #去掉低频词，简单降维
abc[:] = range(len(abc)) #len(abc)=14322 用0-14322间的整数对每个字按顺序重新赋值，一个整数代表一个字
abc[''] = 0 #添加空字符串用来补全
word_set = set(abc.index)#词典


def doc2num(s, maxlen):  # 构建将文本转化为数字向量的函数,maxlen=100
    s = [i for i in s if i in word_set]
    s = s[:maxlen]  # 截取100词
    return list(abc[s])


all_['doc2num'] = all_[2].apply(lambda s: doc2num(s, maxlen))  ##使用函数将文本转化为数字向量

idx = list(range(len(all_))) #生成实际的索引列表
np.random.shuffle(idx) #根据索引打乱文本顺序
all_ = all_.loc[idx] #重新生成表格

#按keras的输入要求来生成数据
x = np.array(list(all_['doc2num']))
y = np.array(list(all_[3]))
y = y.reshape((-1,1)) #调整标签形状

#建立模型
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen,len(abc))))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

#单个one hot矩阵的大小是maxlen*len(abc)的，100*14322 相较于单字的矩阵要大很多。
#为了方便低内存的PC进行测试，这里使用了生成器的方式来生成one hot矩阵
#仅在调用时才生成one hot矩阵
#可以通过减少batch_size来降低内存使用，但会相应地增加一定的训练时间
batch_size = 32 #8G内存不够用，只能降低batch_size
train_num = 5000

#不足则补全0行，maxlen=100，每个句子表示成一个100*14322的矩阵
gen_matrix = lambda z: np.vstack((np_utils.to_categorical(z, len(abc)), np.zeros((maxlen-len(z), len(abc))) ))
#定义数据生成器函数
def data_generator(data, labels, batch_size):
    batches = [range(batch_size*i, min(len(data), batch_size*(i+1))) for i in range(int(len(data)/batch_size)+1)]
    while True:
        for i in batches:
            xx = np.zeros((maxlen, len(abc)))
            xx, yy = np.array(list(map(gen_matrix, data[i]))), labels[i]
            yield (xx, yy)
model.fit_generator(data_generator(x[:train_num], y[:train_num], batch_size), steps_per_epoch=469, epochs=1)
# model.evaluate_generator(data_generator(x[train_num:], y[train_num:], batch_size), steps = 3)
def predict_one(s): #单个句子的预测函数
    s = gen_matrix(doc2num(list(jieba.cut(s)), maxlen))
    s = s.reshape((1, s.shape[0], s.shape[1]))
    return model.predict_classes(s, verbose=0)[0][0]

comment=pd.read_csv("./datas/data_test.csv", encoding='gbk', delimiter="\t",header=None)
#取一百篇用模型预测
comment['text'] = comment[2]
pre=[]
X_test=comment["text"].values
print("preing...")
for test_x in X_test:
    temp=predict_one(test_x)
    print(test_x)
    print(temp)
    print("=======")
    pre.append(temp)

count=1
with open("baidu_sub_kerar18.csv","w") as f:
    for i in range(len(pre)):
        f.write(str(count)+","+str(int(pre[i]))+"\n")
        count=count+1
print("over")