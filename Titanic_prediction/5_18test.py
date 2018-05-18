import pandas as pd
import numpy as np
import lightgbm as lgb
import pandas as pd
import numpy as np
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
                                                  test_size=0.2,
                                                  random_state=42)

lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'binary_logloss', 'auc'},
    'num_leaves': 5,
    'max_depth': 6,
    'min_data_in_leaf': 450,
    'learning_rate': 0.1,
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
                num_boost_round=4000,
                valid_sets=lgb_eval,
                )

ypred = gbm.predict(X_test)
print(ypred)
for i in range(len(ypred)):
    if ypred[i]>=0.55:
        ypred[i]=1
    else:
        ypred[i]=0
print(ypred)

# 导出特征重要性

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
    if pre[i]>=0.5:
        pre[i]=1
    else:
        pre[i]=0
with open("baidu_sub13.csv","w") as f:
    for i in range(len(pre)):
        f.write(str(i+1)+","+str(int(pre[i]))+"\n")

print("over")

