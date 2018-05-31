import pandas as pd
import numpy as np
import lightgbm as lgb
import pandas as pd
import numpy as np
import copy
import pickle
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn import metrics
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

X_train, X_test, y_train, y_test = train_test_split(dataset_X, dataset_Y,
                                                  test_size=0.02,
                                                  random_state=20)

lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
# print("starting first testing......")
# clf = lgb.LGBMClassifier(
#         boosting_type='gbdt', num_leaves=10, reg_alpha=0.0, reg_lambda=1,
#         max_depth=-1, n_estimators=1500, objective='binary',
#         subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
#         learning_rate=0.05, min_child_weight=50, random_state=2018, n_jobs=100
#     )
# clf.fit(X_train, y_train, eval_set=[(X_train, y_train)], eval_metric='auc')
# pre1=clf.predict(X_test)
# acc1=metrics.accuracy_score(y_test,pre1)
# print(acc1)
# print("first over......")
# print(pre1)



params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'binary_logloss', 'auc'},
    'num_leaves': 15,
    'max_depth': -1,
    'min_data_in_leaf': 450,
    'learning_rate': 0.001,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.95,
    'bagging_freq': 5,
    'lambda_l1': 1,
    'lambda_l2': 0.001,  # 越小l2正则程度越高
    'min_gain_to_split': 0.2,
    'verbose': 5,
    'is_unbalance': True
}
print('Start training...')
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=35000,
                valid_sets=lgb_eval,
                )

ypred = gbm.predict(X_test)
print(ypred)
cat=0.3
max=0
remcat=0
for ii in range(500):
    temp=copy.copy(ypred)
    for i in range(len(ypred)):
        if temp[i]>=cat:
            temp[i]=1
        else:
            temp[i]=0
    acc=metrics.accuracy_score(y_test,temp)
    # print("acc="+str(acc)+"|| cat="+str(cat))
    if max<acc:
        max=acc
        remcat=cat
    cat=cat+0.01
print(max)
print(remcat)

for i in range(len(ypred)):
    if ypred[i]>=0.5:
        ypred[i]=1
    else:
        ypred[i]=0
print(ypred)

# 导出特征重要性
print(ypred)
importance = gbm.feature_importance()
names = gbm.feature_name()
print(names)
print(importance)

acc=metrics.accuracy_score(y_test,ypred)
print(acc)

col_names = ["ID","K1K2驱动信号","电子锁驱动信号","急停信号","门禁信号","THDV-M","THDI-M"]
data = pd.read_csv("data_test.csv",names=col_names)
# print(data.info())

# data["test1"] = data["THDI-M"] * data["THDV-M"]
data["test2"] = data["急停信号"] * data["THDV-M"]
data["test3"] = data["THDI-M"] / data["THDV-M"]
data["test4"] = data["THDI-M"] / data["THDV-M"]

print(data.describe())
datasett_X = data[["K1K2驱动信号","电子锁驱动信号","急停信号","门禁信号","THDV-M","THDI-M","test2","test4"]].as_matrix()


pre = gbm.predict(datasett_X)
# pre=clf.predict(datasett_X)
for i in range(len(pre)):
    if pre[i]>=0.51:
        pre[i]=1
    else:
        pre[i]=0
with open("baidu_sub16.csv","w") as f:
    for i in range(len(pre)):
        f.write(str(i+1)+","+str(int(pre[i]))+"\n")

print(acc)
print("over")
