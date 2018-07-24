import jieba
import pandas as pd
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Embedding
from keras.layers import LSTM
datas = pd.read_csv("./datas/data_train.csv", encoding='gbk', delimiter="\t",header=None)
all_ = datas.fillna("无")
all_["word"] = all_[2].apply(lambda s: list(jieba.cut(s,HMM=True)))  ###<span style="color:#ff0000;">语料分词</span>
print(all_)
maxlen = 100  # 截断字数
min_count = 5  # 出现次数少于该值的字扔掉。这是最简单的降维方法
content = []  # 词表
for i in all_["word"]:
    content.extend(i)
# for i in range(100):
#     print(content[i])
abc = pd.Series(list(content)).value_counts()  # 统计词频
print("abc1")
print(abc)
abc = abc[abc >= min_count]  # 去掉低频词，简单降维
print("abc2")
print(abc)
abc[:] = list(range(1, len(abc)+1))  # len(abc)=14322 用0-14322间的整数对每个字按顺序重新赋值，一个整数代表一个字
abc[''] = 0  # 添加空字符串用来补全
print(abc)
word_set = set(abc.index)  # 词典
print("word_set")
print(word_set)
def doc2num(s, maxlen):  # 构建将文本转化为数字向量的函数,maxlen=100
    s = [i for i in s if i in word_set]
    s = s[:maxlen] + [''] * max(0, maxlen - len(s))  # 100词，多截少补空
    return list(abc[s])

all_["doc2num"] = all_["word"].apply(lambda s: doc2num(s, maxlen))  ##使用函数将文本转化为数字向量

print("all")
print(all_)

x = np.array(list(all_["doc2num"]))
y = np.array(list(all_[3]))
y = y.reshape((-1,1)) #调整标签形状
print(x)
print(y)

model = Sequential()
model.add(Embedding(len(abc), 256, input_length=maxlen)) # embdding层，生成字向量
model.add(LSTM(128))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
batch_size = 1000 #批尺寸
train_num = 40000 #训练集数量
model.fit(x[:train_num], y[:train_num], batch_size = batch_size, epochs=1)
model.evaluate(x[train_num:], y[train_num:], batch_size = batch_size)


def predict_one(s): #单个句子的预测函数
    s = np.array(doc2num(list(jieba.cut(s)), maxlen)) #分词
    s = s.reshape((1, s.shape[0]))
    return model.predict_classes(s, verbose=0)[0][0]
comment=pd.read_csv("./datas/data_test.csv", encoding='gbk', delimiter="\t",header=None)
#取一百篇用模型预测
comment['text'] = comment[2]
aaaa = pd.DataFrame(comment[2])
aaaa["result"]=aaaa.text.apply(lambda s: predict_one(s))

pre=aaaa["result"].values
count=1
with open("baidu_sub_kerar10.csv","w") as f:
    for i in range(len(pre)):
        f.write(str(count)+","+str(int(pre[i]))+"\n")
        count=count+1












