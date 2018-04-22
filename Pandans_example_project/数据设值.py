import pandas as pd
import numpy as np
dates = pd.date_range('20130101', periods=6)
df = pd.DataFrame(np.arange(24).reshape((6,4)),index=dates, columns=['A','B','C','D'])
print(df)
#两种索引方式赋值
df.iloc[2,2]=111
df.loc['20130101','B']=222
print(df)

#筛选数据
# df[df.A>4]=0  #将A大于4的这一整行都设为0
# df.A[df.A>4]=0 #将A大于4的这个数设为0

