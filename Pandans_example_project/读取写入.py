import pandas as pd

#读取数据
data=pd.read_csv('student.csv')

#存储
data.to_csv('test.csv')