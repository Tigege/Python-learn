with open ("baidu_analys(1).csv","w") as f:
    for i in range(35157):
        f.write(str(i+1)+","+str(2)+"\n")
print("over")