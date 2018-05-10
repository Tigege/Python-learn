from sklearn import svm
from sklearn import datasets
from sklearn.externals import joblib
clf=svm.SVC()
iris=datasets.load_iris()
X,y=iris.data,iris.target
clf.fit(X,y)

#save 保存
joblib.dump(clf,"save.pkl")

#restore  读取
clf3=joblib.load("save.pkl")
prediction=clf3.predict(X)
print(prediction)