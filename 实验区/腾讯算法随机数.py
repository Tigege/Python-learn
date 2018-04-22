from random import *
s= open(r"C:\Users\Administrator\Desktop\submission.csv",'w',encoding='utf-8')

s.close()
num=0

with open(r"C:\Users\Administrator\Desktop\test1.csv",'r') as f:
    with open(r"C:\Users\Administrator\Desktop\submission.csv", 'a',encoding='utf-8') as s:
        for i in f:
            num=num+1

            l = i.strip()
            ll=l.split(',')
          #  print(ll)

            if num==1:
                pass
            else:
                number=round(uniform(0.3,0.75),8)+round(uniform(0,0.25),8)

                s.write(str(ll[0])+','+str(ll[1]) + ',' + str(number) +'\n')
print("over")



