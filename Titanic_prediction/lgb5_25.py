import lightgbm as lgb
from sklearn.model_selection import train_test_split
import return_data2
dataset_X,dataset_Y=return_data2.return_tarin_data()
acc=[]
randoms=[25]
for random in randoms:
    X_train, X_test, y_train, y_test = train_test_split(dataset_X, dataset_Y,
                                                      test_size=0,
                                                      random_state=random)

    print("LGB test")
    clf = lgb.LGBMClassifier(
        boosting_type='gbdt', num_leaves=55, reg_alpha=0.0, reg_lambda=1,
        max_depth=15, n_estimators=6000, objective='binary',
        subsample=0.8, colsample_bytree=0.8, subsample_freq=1,
        learning_rate=0.06, min_child_weight=1, random_state=20, n_jobs=4
    )
    clf.fit(X_train, y_train)
    # print(clf.feature_importances_)
    testdata=return_data2.return_test_data()
    pre=clf.predict(testdata)
    with open("baidu_sub25.csv", "w") as f:
        for i in range(len(pre)):
            f.write(str(i + 1) + "," + str(pre[i]) + "\n")

    print("over")
