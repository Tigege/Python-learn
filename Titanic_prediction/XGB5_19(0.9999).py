import xgboost as xgb
from xgboost.sklearn import XGBClassifier
import return_data
from sklearn.model_selection import train_test_split
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")
dataset_X,dataset_Y=return_data.return_tarin_data()

X_train, X_test, y_train, y_test = train_test_split(dataset_X, dataset_Y,
                                                  test_size=0.2,
                                                  random_state=45)

xgb_val = xgb.DMatrix(X_test, label=y_test)
xgb_train = xgb.DMatrix(X_train, label=y_train)


xgb1 = XGBClassifier(
 learning_rate =0.05,
 n_estimators=2800,
 max_depth=5,
 min_child_weight=1,
 gamma=0.21,
 subsample=0.8,
 colsample_bytree=0.75,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
print("fiting")
xgb1.fit(X_train,y_train)
pre=xgb1.predict(X_test)
print(pre)
print(y_test)
acc=metrics.accuracy_score(y_test,pre)
print(acc)

X_pre=return_data.return_test_data()
y_pre=xgb1.predict(X_pre)

with open("baidu_sub21.csv","w") as f:
    for i in range(len(y_pre)):
        f.write(str(i+1)+","+str(int(y_pre[i]))+"\n")
print("pre over..")
