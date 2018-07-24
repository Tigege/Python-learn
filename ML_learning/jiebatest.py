import jieba
import pandas as pd
import numpy as np

datas = pd.read_csv("./datas/data_train.csv", encoding='gbk', delimiter="\t",header=None)
all_ = datas.fillna("无")
all_["test"]=all_[2]
print(all_)

print("values")
print(all_["test"].values)
pre=[1,2,3,4,5,6,7,8,90]
count=1
with open("baidu_sub_kerar18.csv","w") as f:
    for i in range(len(pre)):
        f.write(str(count)+","+str(int(pre[i]))+"\n")
        count=count+1
print("over")
