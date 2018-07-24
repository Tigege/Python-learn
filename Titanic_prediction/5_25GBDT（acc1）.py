from sklearn.ensemble import GradientBoostingClassifier
from sklearn import cross_validation, metrics
from sklearn.model_selection import GridSearchCV
import return_data2
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")
import time
from sklearn.externals import joblib
dataset_X,dataset_Y=return_data2.return_tarin_data()
X_train, X_test, y_train, y_test = train_test_split(dataset_X, dataset_Y,
                                                      test_size=0.2,
                                                      random_state=21)
clf = GradientBoostingClassifier(
    learning_rate=0.1, n_estimators=1500, min_samples_split=350,
    min_samples_leaf=20, max_depth=8, max_features="auto", subsample=0.8, random_state=0
)
print("fiting....")
clf.fit(X_train,y_train)
joblib.dump(clf, "baidu_train_modelGDBT.m")
X_pre=return_data2.return_test_data()
y_pre=clf.predict(X_pre)

with open("baidu_subGDBT.csv","w") as f:
    for i in range(len(y_pre)):
        f.write(str(i+1)+","+str(int(y_pre[i]))+"\n")
print("pre over..")


y_pred = clf.predict(X_test)
acc = metrics.accuracy_score(y_test, y_pred)
print("test Acc:",acc)
print("show pre:\n",y_pred)