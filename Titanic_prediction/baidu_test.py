import pandas as pd
import sklearn.preprocessing as preprocessing

def make_data():
    col_names = ["ID","K1K2驱动信号","电子锁驱动信号","急停信号","门禁信号","THDV-M","THDI-M","label"]
    data = pd.read_csv("data_train.csv",names=col_names)
    # print(data.info())

    data = data.fillna(0)
    data['label2'] = data['label'].apply(lambda s: 1 - s)
    data["test1"]=data["THDI-M"]*data["THDV-M"]
    data["test2"]=data["急停信号"]*data["THDV-M"]
    data["test3"]=data["急停信号"]*data["THDV-M"]*data["THDI-M"]
    data["test4"]=data["THDI-M"]/data["THDV-M"]

    scaler=preprocessing.StandardScaler()
    lists=["K1K2驱动信号", "电子锁驱动信号", "急停信号", "门禁信号", "THDV-M", "THDI-M","test1","test2","test3","test4"]
    for list in lists:
        data[list] = scaler.fit_transform(data[[list]])
    '''
    data["test1"] = scaler.fit_transform(data[["test1"]])
    data["test2"] = scaler.fit_transform(data[["test2"]])
    data["K1K2驱动信号"]=scaler.fit_transform(data[["K1K2驱动信号"]])
    data["电子锁驱动信号"] = scaler.fit_transform(data[["电子锁驱动信号"]])
    data["急停信号"] = scaler.fit_transform(data[["急停信号"]])
    data["门禁信号"] = scaler.fit_transform(data[["门禁信号"]])
    data["THDV-M"] = scaler.fit_transform(data[["THDV-M"]])
    data["THDI-M"] = scaler.fit_transform(data[["THDI-M"]])
    '''
    print(data.describe())
    print(data)
    #print(data[["K1K2驱动信号","电子锁驱动信号","急停信号","门禁信号","THDV-M","THDI-M","label","label2"]])
    return data
def read_data():
    col_names = ["ID", "K1K2驱动信号", "电子锁驱动信号", "急停信号", "门禁信号", "THDV-M", "THDI-M",]
    data = pd.read_csv("data_test.csv", names=col_names)
   # print(data.info())
    data = data.fillna(0)
    data["test1"] = data["THDI-M"] * data["THDV-M"]
    data["test2"] = data["急停信号"] * data["THDV-M"]
    data["test3"] = data["急停信号"] * data["THDV-M"] * data["THDI-M"]
    data["test4"] = data["THDI-M"] / data["THDV-M"]
    scaler = preprocessing.StandardScaler()
    lists=["K1K2驱动信号", "电子锁驱动信号", "急停信号", "门禁信号", "THDV-M", "THDI-M","test1","test2","test3","test4"]
    for list in lists:
        data[list] = scaler.fit_transform(data[[list]])

    '''
    data["test1"] = scaler.fit_transform(data[["test1"]])
    data["test2"] = scaler.fit_transform(data[["test2"]])
    data["K1K2驱动信号"] = scaler.fit_transform(data[["K1K2驱动信号"]])
    data["电子锁驱动信号"] = scaler.fit_transform(data[["电子锁驱动信号"]])
    data["急停信号"] = scaler.fit_transform(data[["急停信号"]])
    data["门禁信号"] = scaler.fit_transform(data[["门禁信号"]])
    data["THDV-M"] = scaler.fit_transform(data[["THDV-M"]])
    data["THDI-M"] = scaler.fit_transform(data[["THDI-M"]])
    '''
    return data[["K1K2驱动信号", "电子锁驱动信号", "急停信号", "门禁信号", "THDV-M", "THDI-M","test1","test2","test3","test4"]],data["ID"]
if __name__=="__main__":

    make_data()
