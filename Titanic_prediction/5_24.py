import xgboost as xgb
from xgboost.sklearn import XGBClassifier
import return_data2
from sklearn.model_selection import train_test_split
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")
dataset_X,dataset_Y=return_data2.return_tarin_data()

randoms=[43,21,54,67,78]
for random in randoms:
    X_train, X_test, y_train, y_test = train_test_split(dataset_X, dataset_Y,
                                                      test_size=0.2,
                                                      random_state=random)
    xgb_val = xgb.DMatrix(X_test, label=y_test)
    xgb_train = xgb.DMatrix(X_train, label=y_train)
    xgb1 = XGBClassifier(
     learning_rate =0.05,
     n_estimators=6000,
     max_depth=6,
     min_child_weight=2,
     gamma=0.5,
     subsample=0.7,
     colsample_bytree=0.7,
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




