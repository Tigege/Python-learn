import pandas as pd
import numpy as np
col_names = ["ID", "label"]
col_nameg = ["label"]
data1 = pd.read_csv("GDBT_submission.csv", names=col_nameg)
data2 = pd.read_csv("baidu_subGDBT1.csv", names=col_names)
number1=data1[["label"]].as_matrix()
number1=np.array(number1).reshape(len(number1))

number2=data2[["label"]].as_matrix()
number2=np.array(number2).reshape(len(number2))

eq=0
wr=0
for i in range(len(number1)):
    if number1[i] == number2[i]:
        eq=eq+1
    else:
        wr=wr+1

print("wrong:"+str(wr))