import numpy as np
import pandas as pd
import jieba #结巴分词
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Embedding
from keras.layers import LSTM

pos = pd.read_excel('pos.xls',header=None,index=None)#10677篇
pos['label'] = 1
neg = pd.read_excel('neg.xls',header=None,index=None)#10428篇
neg['label'] = 0
all_ = pos.append(neg, ignore_index=True)
all_['words'] = all_[0].apply(lambda s: list(jieba.cut(s))) #分词
print(all_)
maxlen = 100 #截断字数
min_count = 5 #出现次数少于该值的字扔掉。这是最简单的降维方法

content = []           #词表
for i in all_['words']:
    content.extend(i)
print("content")
print(content)
abc = pd.Series(content).value_counts() #统计词频
abc = abc[abc >= min_count] #去掉低频词，简单降维
abc[:] = list(range(1, len(abc)+1)) #用0-14323间的整数对每个字按顺序重新赋值，一个整数代表一个字
abc[''] = 0 #添加空字符串用来补全
word_set = set(abc.index)  #词典
print("word_set")
print(word_set)
print("abc")
print(abc)



def doc2num(s, maxlen):  # 构建将文本转化为数字向量的函数,maxlen=100
    s = [i for i in s if i in word_set]
    s = s[:maxlen] + [''] * max(0, maxlen - len(s))  # 100词，多截少补空
    return list(abc[s])


all_['doc2num'] = all_['words'].apply(lambda s: doc2num(s, maxlen))  ##使用函数将文本转化为数字向量

idx = list(range(len(all_))) #生成实际的索引列表
np.random.shuffle(idx) #根据索引打乱文本顺序
all_ = all_.loc[idx] #重新生成表格

#按keras的输入要求来生成数据
x = np.array(list(all_['doc2num']))
y = np.array(list(all_['label']))
y = y.reshape((-1,1)) #调整标签形状
print("xxxx")
print(x)
print("yyyy")
print(y)

#建立模型
model = Sequential()
model.add(Embedding(len(abc), 256, input_length=maxlen)) # embdding层，生成字向量
model.add(LSTM(128))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

batch_size = 128 #批尺寸
train_num = 15000 #训练集数量

model.fit(x[:train_num], y[:train_num], batch_size = batch_size, epochs=1)

print(model.evaluate(x[train_num:], y[train_num:], batch_size = batch_size))

def predict_one(s): #单个句子的预测函数
    s = np.array(doc2num(list(jieba.cut(s)), maxlen)) #分词
    s = s.reshape((1, s.shape[0]))
    return model.predict_classes(s, verbose=0)[0][0]



'''
comment = pd.read_excel('E:/Coding/comment/sum.xls')
comment = comment[comment['rateContent'].notnull()]
comment['text'] = comment['rateContent']

aaaa = pd.DataFrame(comment['text'][500:600])
aaaa['result']=aaaa.text.apply(lambda s: predict_one(s))

'''