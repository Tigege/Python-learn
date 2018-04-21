count=0
with open(r"D:\腾讯微云\preliminary_contest_data\userFeature.data",'r') as f:
    with open(r"D:\腾讯微云\preliminary_contest_data\uF_test1.txt","w") as w:
        for i in f:
            w.write(i)
            count+=1
            if(count==100):
                print('over')
                break