import pandas as pd
import sklearn.preprocessing as preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from abupy import AbuML
from sklearn.metrics import r2_score
col_names = ["ID","K1K2驱动信号","电子锁驱动信号","急停信号","门禁信号","THDV-M","THDI-M","label"]
data = pd.read_csv("data_train.csv",names=col_names)
# print(data.info())
print(data.describe())
dataset_X = data[["K1K2驱动信号","电子锁驱动信号","急停信号","门禁信号","THDV-M","THDI-M"]].as_matrix()
dataset_Y = data[["label"]].as_matrix()
X_train, X_val, y_train, y_val = train_test_split(dataset_X, dataset_Y,
                                                  test_size=0.2,
                                                  random_state=42)

scaler = preprocessing.StandardScaler()
lists=["K1K2驱动信号", "电子锁驱动信号", "急停信号", "门禁信号", "THDV-M", "THDI-M"]
for list in lists:
    data[list] = scaler.fit_transform(data[[list]])

boston=AbuML(X_train,y_train)
boston.estimator.polynomial_regression(degree=2)
reg=boston.fit()

y_pred=reg.predict(X_val)
print(r2_score(y_val,y_pred))

