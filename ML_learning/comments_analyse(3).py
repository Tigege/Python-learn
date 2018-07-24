import comments_return_2
from sklearn.ensemble import GradientBoostingClassifier
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
data,dataset_Y=comments_return_2.test_data()
p1,p2,p3,p4=comments_return_2.test_tetsdata()
# print(data)
# print(dataset_Y)
X_train, X_test, y_train, y_test = train_test_split(data, dataset_Y,
                                                      test_size=0.3,
                                                      random_state=21)
feature_extraction=TfidfVectorizer()
X_train=feature_extraction.fit_transform(X_train)

svc =  GradientBoostingClassifier(
    learning_rate=0.15, n_estimators=1500, min_samples_split=350,
    min_samples_leaf=20, max_depth=8, max_features="auto", subsample=0.8, random_state=0
)
print("fiting")
svc.fit(X_train,y_train)

X_testt=feature_extraction.transform(X_test)

pre=svc.predict(X_testt)
acc=metrics.accuracy_score(y_test,pre)
print(acc)

'''
X_test1=feature_extraction.transform(p1)
del p1
print("pre1..")
pre1=svc.predict(X_test1)
X_test2=feature_extraction.transform(p2)
del p2
print("pre2..")
pre2=svc.predict(X_test2)
X_test3=feature_extraction.transform(p3)
del p3
print("pre3..")
pre3=svc.predict(X_test3)
X_test4=feature_extraction.transform(p4)
del p4
pre4=svc.predict(X_test4)

# print(X_train.shape)
# print(X_test1.shape)
# print(X_test2.shape)
# print(X_test3.shape)
# print(X_test4.shape)

print("sadf")
print(pre1)
print(pre2)
print(pre3)
print(pre4)

# acc = metrics.accuracy_score(y_test, pre)
# print(acc)
count=1
with open("baidu_sub11.csv","w") as f:
    for i in range(len(pre1)):
        f.write(str(count)+","+str(int(pre1[i]))+"\n")
        count=count+1
    for i in range(len(pre2)):
        f.write(str(count) + "," + str(int(pre2[i])) + "\n")
        count = count + 1
    for i in range(len(pre3)):
        f.write(str(count) + "," + str(int(pre3[i])) + "\n")
        count = count + 1
    for i in range(len(pre4)):
        f.write(str(count) + "," + str(int(pre4[i])) + "\n")
        count = count + 1
print("pre over..")
'''
