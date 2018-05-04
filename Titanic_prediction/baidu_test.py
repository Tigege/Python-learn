import pandas as pd


def make_data():
    col_names = ["ID","K1K2驱动信号","电子锁驱动信号","急停信号","门禁信号","THDV-M","THDI-M","label"]
    data = pd.read_csv("data_train.csv",names=col_names)
    print(data.info())
    print(data.describe())
    data = data.fillna(0)
    data['label2'] = data['label'].apply(lambda s: 1 - s)
    # print(data[["K1K2驱动信号","电子锁驱动信号","急停信号","门禁信号","THDV-M","THDI-M","label","label2"]])
    return data[["K1K2驱动信号","电子锁驱动信号","急停信号","门禁信号","THDV-M","THDI-M","label","label2"]]
def read_data():
    col_names = ["ID", "K1K2驱动信号", "电子锁驱动信号", "急停信号", "门禁信号", "THDV-M", "THDI-M"]
    data = pd.read_csv("data_test.csv", names=col_names)
   # print(data.info())
    data = data.fillna(0)
    return data[["K1K2驱动信号", "电子锁驱动信号", "急停信号", "门禁信号", "THDV-M", "THDI-M"]],data["ID"]
if __name__=="__main__":

    make_data()
