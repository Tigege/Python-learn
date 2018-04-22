import pandas as pd
import numpy as np
'''
s = pd.Series([1,3,6,np.nan,44,1])
print(s)
0     1.0
1     3.0
2     6.0
3     NaN
4    44.0
5     1.0
dtype: float64
'''
'''
dates = pd.date_range('20160101',periods=6)
df = pd.DataFrame(np.random.randn(6,4),index=dates,columns=['a','b','c','d'])
print(df)
'''
dates = pd.date_range('20130101', periods=6)
df = pd.DataFrame(np.arange(24).reshape((6,4)),index=dates, columns=['A','B','C','D'])
print(df)
print('-------------')
print(df.A) #print(da['A'])

print('-------------')
print(df[0:3])  #print(df['20130101':'20130104'])

#数据选择
#纯标签筛选
print(df.loc['20130102'])  #选择  20130102这一行数据
print(df.loc[:,['A','B']]) #选择 ：所有的行  AB这两列
print(df.loc['20130102',['A','B']])

#纯数字筛选
print(df.iloc[3])   #显示第索引是三的
print(df.iloc[3,1]) #显示第三行的第二个数  （下标从0开始）
print(df.iloc[1:3,1:3])  #切片显示

#混合筛选
print(df.ix[:3,['A','B']])   #0到3行  AB两列

#筛选选择
print(df[df.A>8])  #选择A这一列大于8的所有行

