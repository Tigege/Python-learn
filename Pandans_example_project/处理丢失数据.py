import numpy as np
import pandas as pd

dates = pd.date_range('20130101', periods=6)
df = pd.DataFrame(np.arange(24).reshape((6,4)),index=dates, columns=['A','B','C','D'])
df.iloc[0,1] = np.nan
df.iloc[0,2]=np.nan
df.iloc[1,2] = np.nan

print(df)

print(df.dropna(axis=0,how='any')) #0代表行 1代表列  any 只要有一个空，那就删除整行   all是全部是空

df.fillna(value=0)  #将NAN的值用给定值代替

df.isnull()

# True
# 检测在数据中是否存在 NaN, 如果存在就返回 True:
np.any(df.isnull()) == True
