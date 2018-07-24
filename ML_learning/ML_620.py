import numpy as np
import pandas as pd
import jieba #结巴分词
import comments_return_2
import pandas as pd
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
import numpy as np
from sklearn import cross_validation, metrics
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import gc
import lightgbm as lgb
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Embedding
from keras.layers import LSTM
import warnings
warnings.filterwarnings("ignore")
datas = pd.read_csv("./datas/data_train.csv", encoding='gbk', delimiter="\t",header=None)
all_ = datas.fillna("无")
all_["word"] = all_[2].apply(lambda s: list(jieba.cut(s)))  ###<span style="color:#ff0000;">语料分词</span>
print(all_)
maxlen = 100 #截断字数
min_count = 3 #出现次数少于该值的字扔掉。这是最简单的降维方法

content = []           #词表
for i in all_['word']:
    content.extend(i)

abc = pd.Series(content).value_counts() #统计词频
abc = abc[abc >= min_count] #去掉低频词，简单降维
abc[:] = list(range(1, len(abc)+1)) #用0-14323间的整数对每个字按顺序重新赋值，一个整数代表一个字
abc[''] = 0 #添加空字符串用来补全
word_set = set(abc.index)  #词典


def doc2num(s, maxlen):  # 构建将文本转化为数字向量的函数,maxlen=100
    s = [i for i in s if i in word_set]
    s = s[:maxlen] + [''] * max(0, maxlen - len(s))  # 100词，多截少补空
    return list(abc[s])


all_['doc2num'] = all_['word'].apply(lambda s: doc2num(s, maxlen))  ##使用函数将文本转化为数字向量

idx = list(range(len(all_))) #生成实际的索引列表
np.random.shuffle(idx) #根据索引打乱文本顺序
all_ = all_.loc[idx] #重新生成表格

#按keras的输入要求来生成数据
x = np.array(list(all_['doc2num']))
y = np.array(list(all_[3]))
y = y.reshape(len(y)) #调整标签形状

print("XXXXX")
print(x)

print("yyyyy")
print(y)
X_train, X_test, y_train, y_test = train_test_split(x, y,
                                                      test_size=0.3,
                                                      random_state=21)



clf = XGBClassifier(
 learning_rate =0.5,
 n_estimators=2000,
 max_depth=15,
 min_child_weight=5,
 gamma=0.1,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
print("fiting")
clf.fit(X_train,y_train)

pre=clf.predict(X_test)
print("pre")
print(pre)

print("ACC:")
acc = metrics.accuracy_score(y_test, pre)
print(acc)



'''
#建立模型
model = Sequential()
model.add(Embedding(len(abc), 256, input_length=maxlen)) # embdding层，生成字向量
model.add(LSTM(128))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
batch_size = 128 #批尺寸
train_num = 15000 #训练集数量

model.fit(x[:train_num], y[:train_num], batch_size = batch_size, epochs=1)

model.evaluate(x[train_num:], y[train_num:], batch_size = batch_size)
'''
def predict_one(s): #单个句子的预测函数
    s = np.array(doc2num(list(jieba.cut(s)), maxlen)) #分词
    s = s.reshape((1, s.shape[0]))
    print("+++++++")
    print("in _ones")
    print(s)
    print("++++++++")
    return clf.predict(s)


'''
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
'''