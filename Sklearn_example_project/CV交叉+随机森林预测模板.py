from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing as preprocessing
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

col_names = ["ID","K1K2驱动信号","电子锁驱动信号","急停信号","门禁信号","THDV-M","THDI-M","label"]
data = pd.read_csv("data_train.csv",names=col_names)

data["test2"] = data["急停信号"] * data["THDV-M"]
data["test4"] = data["THDI-M"] / data["THDV-M"]

dataset_X = data[["K1K2驱动信号","电子锁驱动信号","急停信号","门禁信号","THDV-M","THDI-M","test2","test4"]].as_matrix()
dataset_Y = data[["label"]].as_matrix()



X_train, X_test, y_train, y_test = train_test_split(dataset_X, dataset_Y,
                                                  test_size=0.2,
                                                  random_state=42)

alphas=[10,15,20,25,30]
test_sores=[]
for alpha in alphas:
    clf = ExtraTreesClassifier(n_estimators=alpha, max_depth=None,
        min_samples_split=3, random_state=1)
    sore=cross_val_score(clf,X_train,
                                        y_train.ravel(),cv=5,
                                        scoring='accuracy')
    test_sores.append(np.mean(sore))
    print("testing",str(alpha))
print(test_sores)
plt.figure()
plt.plot(alphas,test_sores, color='blue')
plt.scatter(alphas,test_sores,s=75,c="red",alpha=0.5)
plt.show()