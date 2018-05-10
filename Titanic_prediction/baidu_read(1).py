import pandas as pd
import numpy as np
import sklearn.preprocessing as preprocessing
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

col_names = ["ID","K1K2驱动信号","电子锁驱动信号","急停信号","门禁信号","THDV-M","THDI-M","label"]
data = pd.read_csv("data_train.csv",names=col_names)
# print(data.info())

# data["test1"] = data["THDI-M"] * data["THDV-M"]
data["test2"] = data["急停信号"] * data["THDV-M"]
data["test3"] = data["THDI-M"] / data["THDV-M"]
data["test4"] = data["THDI-M"] / data["THDV-M"]


print(data.describe())
# scaler = preprocessing.StandardScaler()
# lists=["K1K2驱动信号", "电子锁驱动信号", "急停信号", "门禁信号", "THDV-M", "THDI-M","test1","test2","test3","test4"]
# for list in lists:
#     data[list] = scaler.fit_transform(data[[list]])

dataset_X = data[["K1K2驱动信号","电子锁驱动信号","急停信号","门禁信号","THDV-M","THDI-M","test2","test4"]].as_matrix()
dataset_Y = data[["label"]].as_matrix()

X_train, X_test, y_train, y_test = train_test_split(dataset_X, dataset_Y,
                                                  test_size=0.2,
                                                  random_state=42)
# print(dataset_X)
# print(dataset_Y)
# print("X_train")
# print(preprocessing.scale(X_train))
# print("Y_train is :")
# print(y_train)
# print("ravel is :")
# print(y_train.ravel())
# print("X_train is :")
# print(X_train)
knn=KNeighborsClassifier(n_neighbors=15)
knn.fit(X_train,y_train.ravel())

# col_names = ["ID", "K1K2驱动信号", "电子锁驱动信号", "急停信号", "门禁信号", "THDV-M", "THDI-M",]
# datat = pd.read_csv("data_test.csv", names=col_names)
# dataset_X = datat[["K1K2驱动信号","电子锁驱动信号","急停信号","门禁信号","THDV-M","THDI-M"]].as_matrix()

y_pred_on_test=knn.predict(X_test)
right=0
for i in range(len(y_pred_on_test)):
    if(y_pred_on_test[i]==y_test[i]):
        right=right+1
acc=right/len(y_pred_on_test)
print(acc)


# y_test1=np.array(y_pred_on_test).reshape((len(y_pred_on_test),1))
# acc=knn.score(y_pred_on_test,y_test)
# acc=metrics.accuracy_score(y_test.ravel(),y_test1)

'''
print("+++++++++++++++++++")
print(y_pred_on_test)
with open("baidu_sub5.csv","w") as f:
    for i in range(len(y_pred_on_test)):
        f.write(str(i+1)+","+str(y_pred_on_test[i])+"\n")
'''