import pandas as pd
import numpy as np
from sklearn import preprocessing
def return_tarin_data():
    col_names = ["ID","K1K2驱动信号","电子锁驱动信号","急停信号","门禁信号","THDV-M","THDI-M","label"]
    data = pd.read_csv("data_train.csv",names=col_names)
    # print(data.info())
    data["test1"] = data["THDI-M"] * data["THDV-M"]
    data["test2"] = data["急停信号"] * data["THDV-M"]
    data["test3"] = data["THDI-M"] / data["THDV-M"]
    data["test4"] = data["THDI-M"] / (data["急停信号"]*data["THDV-M"])
    scaler = preprocessing.StandardScaler()
    # lists = ["K1K2驱动信号", "电子锁驱动信号", "急停信号", "门禁信号", "THDV-M", "THDI-M"]
    # for list in lists:
    #     data[list] = scaler.fit_transform(data[[list]])

    # print(data.describe())
    dataset_X = data[["K1K2驱动信号","电子锁驱动信号","急停信号","门禁信号","THDV-M","THDI-M"]].as_matrix()
    dataset_Y = data[["label"]].as_matrix()
    dataset_Y=np.array(dataset_Y).reshape(len(dataset_Y))
    # print(dataset_X)
    # print(dataset_Y)
    return dataset_X,dataset_Y

def return_test_data():
    col_names = ["ID", "K1K2驱动信号", "电子锁驱动信号", "急停信号", "门禁信号", "THDV-M", "THDI-M"]
    data = pd.read_csv("data_test.csv", names=col_names)
    # print(data.info())
    # data["test1"] = data["THDI-M"] * data["THDV-M"]
    data["test2"] = data["急停信号"] * data["THDV-M"]
    data["test3"] = data["THDI-M"] / data["THDV-M"]
    data["test4"] = data["THDI-M"] / (data["急停信号"] * data["THDV-M"])
    # print(data.describe())
    scaler = preprocessing.StandardScaler()
    # lists = ["K1K2驱动信号", "电子锁驱动信号", "急停信号", "门禁信号", "THDV-M", "THDI-M"]
    # for list in lists:
    #     data[list] = scaler.fit_transform(data[[list]])
    datasett_X = data[["K1K2驱动信号", "电子锁驱动信号", "急停信号", "门禁信号", "THDV-M", "THDI-M"]].as_matrix()
    return datasett_X