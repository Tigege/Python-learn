import numpy as np
import pandas as pd
import jieba.analyse
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Embedding
from keras.layers import LSTM
import read_stop_worlds
col_names = ["ID","类型","评论内容","标签"]
data = pd.read_csv("./datas/data_train.csv",names=col_names,encoding='ANSI',sep='\t')
data1 = pd.read_csv("./datas/data_test.csv",names=col_names,encoding='ANSI',sep='\t')

data = np.array(data)
data_train = data[:,2:3]
print(data_train)
data_label = data[:,3]

data1 = np.array(data1)
data_train1 = data1[:,2:3]
data_label1 = data1[:,3]

data_train = np.delete(data_train,27027,0)
data_train = np.delete(data_train,69880,0)
data_label = np.delete(data_label,27027,0)
data_label = np.delete(data_label,69880,0)

data_train1 = np.delete(data_train1,27186,0)

data_train = pd.DataFrame(data_train)
#
data_train1 = pd.DataFrame(data_train1)
#
data_train['words'] = data_train[0].apply(lambda s: list(jieba.cut(s)))
data_train1['words'] = data_train1[0].apply(lambda s: list(jieba.cut(s)))
stopworldss=read_stop_worlds.return_stop_wrolds()
stopworldss.append(",")
stopworldss.append("，")
content_temp = []
content=[]
for i in data_train['words']:
    content_temp.extend(i)
for i in content_temp:
    if i not in stopworldss:
        content.append(i)

maxlen = 150 #截断字数
min_count = 0#去掉低频词，简单降维 #出现次数少于该值的字扔掉。这是最简单的降维方法

abc = pd.Series(content).value_counts() #统计词频
print(abc)
abc = abc[abc >= min_count]
abc[:] = list(range(1, len(abc)+1)) #用0-14323间的整数对每个字按顺序重新赋值，一个整数代表一个字
abc[''] = 0 #添加空字符串用来补全
word_set = set(abc.index)  #词典

def doc2num(s, maxlen):                  #构建将文本转化为数字向量的函数,maxlen=100
    s = [i for i in s if i in word_set]
    s = s[:maxlen] + ['']*max(0, maxlen-len(s)) #100词，多截少补空
    return list(abc[s])


data_train['doc2num'] = data_train['words'].apply(lambda s: doc2num(s, maxlen)) ##使用函数将文本转化为数字向量
data_train = np.array(list(data_train['doc2num']))
data_train1['doc2num'] = data_train1['words'].apply(lambda s: doc2num(s, maxlen))
data_train1 = np.array(list(data_train1['doc2num']))
data_train[27027]=0
data_train[69880]=0
d = []
for i in range(len(data_label)):
    if data_label[i] == 0:
        d.append([1, 0, 0])
    if data_label[i] == 1:
        d.append([0, 1, 0])
    if data_label[i] == 2:
        d.append([0, 0, 1])
d = np.array(d)
#
train_x, test_x, train_y, test_y = train_test_split(data_train, d, test_size=0.01, random_state=2018)
model = Sequential()
model.add(Embedding(len(abc), 256, input_length=maxlen))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(3))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_x, train_y, batch_size=128, epochs=10)
model.evaluate(test_x, test_y, batch_size=128)
y = model.predict(data_train1)
ans = []
for i in range(len(y)):
    if i < 27186:
        ans.append(np.argmax(y[i]))
    elif i == 27186:
        ans.append(1)
        ans.append(np.argmax(y[i]))
    else:
        ans.append(np.argmax(y[i]))
# print(ans)
# print(len(ans))
# print(len(y))
# print(data_train1)

file = open('./submission711.csv', 'w')
for i in range(len(ans)):
    file.writelines(str(i+1) + ',' + str(ans[i]) + '\n')
file.close()
print("over")