from sklearn.externals import joblib
import return_data2

clf = joblib.load("baidu_train_modelGDBT.m")
X_pre=return_data2.return_test_data()
y_pre=clf.predict(X_pre)

with open("baidu_subGDBT1.csv","w") as f:
    for i in range(len(y_pre)):
        f.write(str(i+1)+","+str(int(y_pre[i]))+"\n")
print("pre over..")