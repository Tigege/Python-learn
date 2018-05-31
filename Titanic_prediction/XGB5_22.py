import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import pandas as pd
import return_data
from xgboost import plot_importance
from matplotlib import pyplot as plt
import warnings
import numpy as np
from sklearn.model_selection import cross_val_score
warnings.filterwarnings("ignore")
def write_to_file(lists):
    with open("baidu_sub22.csv", "w") as f:
        for i in range(len(lists)):
            f.write(str(i + 1) + "," + str(int(lists[i])) + "\n")
    print("pre over..")

xgb1 = XGBClassifier(
     learning_rate =0.06,
     n_estimators=2700,
     max_depth=5,
     min_child_weight=1,
     gamma=0.21,
     subsample=0.8,
     colsample_bytree=0.75,
     objective= 'binary:logistic',
     nthread=4,
     scale_pos_weight=1,
     seed=27)
test_sores=[]
X_train,y_train=return_data.return_tarin_data()
sore=cross_val_score(xgb1,X_train,y_train,cv=5,scoring='neg_mean_squared_error')
test_sores.append(np.mean(sore))

print(test_sores)