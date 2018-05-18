from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
import sklearn.preprocessing as preprocessing
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from  sklearn.ensemble  import  GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score

col_names = ["ID","K1K2驱动信号","电子锁驱动信号","急停信号","门禁信号","THDV-M","THDI-M","label"]
data = pd.read_csv("data_train.csv",names=col_names)
# print(data.info())

data["test1"] = data["THDI-M"] * data["THDV-M"]
data["test2"] = data["急停信号"] * data["THDV-M"]
data["test3"] = data["THDI-M"] / data["THDV-M"]
data["test4"] = data["THDI-M"] / data["THDV-M"]
print(data.describe())
dataset_X = data[["K1K2驱动信号","电子锁驱动信号","急停信号","门禁信号","THDV-M","THDI-M","test2","test4"]].as_matrix()
dataset_Y = data[["label"]].as_matrix()

dataset_Y=np.array(dataset_Y).reshape(len(dataset_Y))

clf=GradientBoostingClassifier(
  loss='deviance'
, learning_rate=0.1
, n_estimators=100
, subsample=1
, min_samples_split=2
, min_samples_leaf=1
, max_depth=10
, init=None
, random_state=None
, max_features=None
, verbose=0
, max_leaf_nodes=None
, warm_start=False
)
scores=cross_val_score(clf,dataset_X,dataset_Y,cv=5)
print(scores)
# print(clf.score(X_test, y_test))
# pre=clf.predict(X_test)
# right=0
# for i in range(len(pre)):
#     if(pre[i]==y_test[i]):
#         right=right+1
# acc=right/len(pre)
# print("acc=========")
# print(acc)
'''
col_names = ["ID","K1K2驱动信号","电子锁驱动信号","急停信号","门禁信号","THDV-M","THDI-M"]
data = pd.read_csv("data_test.csv",names=col_names)
# print(data.info())

# data["test1"] = data["THDI-M"] * data["THDV-M"]
data["test2"] = data["急停信号"] * data["THDV-M"]
data["test3"] = data["THDI-M"] / data["THDV-M"]
data["test4"] = data["THDI-M"] / data["THDV-M"]

print(data.describe())
datasett_X = data[["K1K2驱动信号","电子锁驱动信号","急停信号","门禁信号","THDV-M","THDI-M","test2","test4"]].as_matrix()

pre=clf.predict(datasett_X)

with open("baidu_sub11.csv","w") as f:
    for i in range(len(pre)):
        f.write(str(i+1)+","+str(pre[i])+"\n")

print("over")
'''