import pandas as pd
import jiebatest
import re
import numpy as np
from sklearn.model_selection import train_test_split

def test_data():
    list=["ID","类型","评论","lable"]
    datas=pd.read_csv("./datas/data_train.csv",encoding='gbk',delimiter="\t",names=list)
    datas = datas.fillna("无")
    # print(datas[["评论"]].head(5))
    # print(datas.describe())
    # print(datas.info())
    # print("-----")

    traincommint=datas[["评论"]]
    worlds=[]
    for i in range(82025):
        worlds.append(traincommint.iloc[i,0])
    l=[]
    for i in range(80000):
        # print(worlds[i])
        # print(worlds[i])
        seg_list = jiebatest.cut(worlds[i], cut_all=False)
        lists="/ ".join(seg_list)  # 全模式
        # print("_____")
        # print(list)
        listss=lists.split("/")
        list=""
        for i in listss:
            list=list+str(i)
        # print(list)
        l.append(list)   # 按照指定字符分割，以列表方式返回
    lables=datas[["lable"]].as_matrix()
    lables = np.array(lables).reshape(len(lables))
    data_y=[]
    for i in range(80000):
        data_y.append(lables[i])
    print(l)
    print(data_y)
    X_train, X_test, y_train, y_test = train_test_split(l, data_y,
                                                        test_size=0.5,
                                                        random_state=21)
    return X_train,y_train


def test_tetsdata():
    list=["ID","类型","评论"]
    datas=pd.read_csv("./datas/data_test.csv",encoding='gbk',delimiter="\t",names=list)
    datas = datas.fillna("无")
    # print(datas[["评论"]].head(5))
    # print(datas.describe())
    # print(datas.info())
    # print("-----")
    # print(len(datas))
    traincommint=datas[["评论"]]
    worlds=[]
    for i in range(len(datas)):
        worlds.append(traincommint.iloc[i,0])
    l=[]
    for i in range(len(datas)):
        seg_list = jiebatest.cut(worlds[i], cut_all=False)
        lists="/ ".join(seg_list)
        listss=lists.split("/")
        list=""
        for i in listss:
            list=list+str(i)
        l.append(list)   # 按照指定字符分割，以列表方式返回
    p1=l[:10000]
    p2=l[10000:20000]
    p3=l[20000:30000]
    p4=l[30000:len(datas)]
    # print(data_y)
    return p1,p2,p3,p4

if __name__=="__main__":
    # test_data()
    test_tetsdata()

