import pandas as pd
import re
import numpy as np
from sklearn.model_selection import train_test_split

import jieba
# print(datas[["评论"]].head(5))
# print(datas.describe())
# print(datas.info())
# print("-----")
def test_data():
    list=["ID","类型","评论","lable"]
    datas=pd.read_csv("./datas/data_train.csv",encoding='gbk',delimiter="\t",names=list)
    datas = datas.fillna("无")
    #分词
    col=datas.iloc[:,2]   #获得评论一整列
    print(col.values)     #
    worlds=col.values
    l=[]
    for i in range(len(worlds)):
        seg_list = jieba.cut(worlds[i],HMM=True)
        lists=" ".join(seg_list)  # 全模式 用空格分隔
        l.append(lists)

    #标签
    lables=datas[["lable"]].as_matrix()
    lables = np.array(lables).reshape(len(lables))
    X_train, X_test, y_train, y_test = train_test_split(l, lables,
                                                        test_size=0,  #小数是代表测试集的比例
                                                        random_state=21)
    return X_train,y_train


def test_tetsdata():
    list=["ID","类型","评论"]
    datas=pd.read_csv("./datas/data_test.csv",encoding='gbk',delimiter="\t",names=list)
    datas = datas.fillna("无")
    col=datas.iloc[:,2]
    worlds = col.values
    l=[]
    for i in range(len(worlds)):
        seg_list = jieba.cut(worlds[i], cut_all=False)
        lists=" ".join(seg_list)
        l.append(lists)   # 按照指定字符分割，以列表方式返回
    p1=l[:10000]
    p2=l[10000:20000]
    p3=l[20000:30000]
    p4=l[30000:len(datas)]
    # print(data_y)
    return p1,p2,p3,p4


if __name__=="__main__":
    test_data()
    # test_tetsdata()

