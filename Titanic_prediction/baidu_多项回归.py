import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn import cross_validation
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

col_names = ["ID","K1K2驱动信号","电子锁驱动信号","急停信号","门禁信号","THDV-M","THDI-M","label"]
data = pd.read_csv("data_train.csv",names=col_names)

dataset_X = data[["K1K2驱动信号","电子锁驱动信号","急停信号","门禁信号","THDV-M","THDI-M"]].as_matrix()
dataset_Y = data[["label"]].as_matrix()


poly_reg = PolynomialFeatures(degree = 1) #degree 就是自变量需要的维度
X_poly = poly_reg.fit_transform(dataset_X)

X_train, X_test, y_train, y_test = train_test_split(X_poly, dataset_Y,
                                                  test_size=0.2,
                                                  random_state=42)

lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_train,y_train.ravel())

pre=lin_reg_2.predict(X_test)
right=0
for i in range(len(pre)):
    if(pre[i]>=0.52):
        pre[i]=1
    else:
        pre[i]=0

for i in range(len(pre)):
    if(pre[i]==y_test[i]):
        right=right+1
acc=right/len(pre)
print(pre)
print("++++++++++")
print(y_test)
print("++++++++++")
print(acc)



