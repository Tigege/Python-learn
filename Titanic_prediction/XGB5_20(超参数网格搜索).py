import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import pandas as pd
import return_data
from xgboost import plot_importance
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings("ignore")
def write_to_file(lists):
    with open("baidu_sub22.csv", "w") as f:
        for i in range(len(lists)):
            f.write(str(i + 1) + "," + str(int(lists[i])) + "\n")
    print("pre over..")


#alg传入XGBOOST,X_train传入训练数据特征信息，Y_train传入训练数据标签信息  X_testdata最后要预测的值
def XGBmodelfit(alg, X_train, Y_train,X_test=None,Y_test=None,X_predictions=None,useTrainCV=True, cv_folds=5, early_stopping_rounds=200):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(X_train, label=Y_train)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='auc', early_stopping_rounds=early_stopping_rounds, show_stdv=False)
        # alg.set_params(n_estimators=cvresult.shape[0])

    #训练模型
    print("fiting")
    alg.fit(X_train, Y_train,eval_metric='auc')

    #预测结果:
    # dtrain_predictions = alg.predict(X_test)  #输出 0 或 1
    # dtrain_predprob = alg.predict_proba(X_test)[:,1]   #输出概率

    #打印报告信息:
    # print("\nModel Report")
    # print("Accuracy  (Train) : %.4g" % metrics.accuracy_score(Y_test, dtrain_predictions))
    # print("AUC Score (Train): %f" % metrics.roc_auc_score(Y_test, dtrain_predictions))
    # print(alg)
    # print("the best:")
    # print(cvresult.shape[0])
    # plot_importance(alg)
    # plt.show()
    print("stating predicting ..")
    test_data=return_data.return_test_data()
    pretest=alg.predict(test_data)
    print("writing>>>")
    write_to_file(pretest)
    # feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    # feat_imp.plot(kind='bar', title='Feature Importances')
    # plt.ylabel('Feature Importance Score')

dataset_X,dataset_Y=return_data.return_tarin_data()

X_train, X_test, y_train, y_test = train_test_split(dataset_X, dataset_Y,
                                                  test_size=0.2,
                                                  random_state=45)

xgb1 = XGBClassifier(
     learning_rate =0.06,
     n_estimators=2700,
     max_depth=6,
     min_child_weight=1,
     gamma=0.21,
     subsample=0.8,
     colsample_bytree=0.75,
     objective= 'binary:logistic',
     nthread=2,
     scale_pos_weight=1,
     seed=25)
xgb1.fit(X_train,y_train)
print("stating predicting ..")
test_data=return_data.return_test_data()
pretest=xgb1.predict(test_data)
print("writing>>>")
write_to_file(pretest)

# XGBmodelfit(xgb1,X_train,y_train,X_test,y_test)


 # gsearch1 = GridSearchCV(estimator = XGBClassifier(
 #       learning_rate =0.1, n_estimators=140, max_depth=9,
 #       min_child_weight=1, gamma=0, subsample=0.8,colsample_bytree=0.8,
 #       objective= 'binary:logistic', nthread=4,scale_pos_weight=1, seed=27),
 #       param_grid=param_grid,cv=5)

param_test1 = {
 'max_depth':range(3,7,2),
 'min_child_weight':range(1,6,2)
}
param_test2 = {
 'max_depth':[4,5,6],
 'min_child_weight':[1,2,3]
}
param_test2b = {
 'min_child_weight':[6,8,10,12]
 }
#[0.0, 0.1, 0.2, 0.3, 0.4]
param_test3 = {
 'gamma':[i/10.0 for i in range(0,5)]
}
param_test3b = {
 'gamma':[0.17,0.18,0.19,0.20,0.21,0.22,0.23,0.24,0.25]
}
param_test4 = {
 'subsample':[i/10.0 for i in range(6,10)],
 'colsample_bytree':[i/10.0 for i in range(6,10)]
}
param_test5 = {
 'subsample':[i/100.0 for i in range(75,90,5)],
 'colsample_bytree':[i/100.0 for i in range(75,90,5)]
}
param_test6 = {
 'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
}
param_test7 = {
 'reg_alpha':[0, 0.001, 0.005, 0.01, 0.05]
}
param_test8 = {
 'seed':[24,25,26,27,28]
}
param_test9 = {
 'learning_rate':[0.04,0.05,0.06]
}
param_test9 = {
 'n_estimators':[2500]
}

gsearch1 = GridSearchCV(estimator =
                        XGBClassifier(
                         learning_rate =0.06,
                         n_estimators=2500,
                         max_depth=6,
                         min_child_weight=1,
                         gamma=0.2,
                         subsample=0.8,
                         colsample_bytree=0.75,
                         objective= 'binary:logistic',
                         nthread=2,
                         scale_pos_weight=1,
                         seed=25),
         param_grid=param_test9,cv=5,verbose=5)

gsearch1.fit(X_train,y_train)
print(gsearch1.best_params_,gsearch1.best_score_)


#
# param_test2 = {
#  'max_depth':[4,5,6],
#  'min_child_weight':[4,5,6]
# }
# param_test2b = {
#  'min_child_weight':[6,8,10,12]
#  }
# #[0.0, 0.1, 0.2, 0.3, 0.4]
# param_test3 = {
#  'gamma':[i/10.0 for i in range(0,5)]
# }
# param_test4 = {
#  'subsample':[i/10.0 for i in range(6,10)],
#  'colsample_bytree':[i/10.0 for i in range(6,10)]
# }
# param_test5 = {
#  'subsample':[i/100.0 for i in range(75,90,5)],
#  'colsample_bytree':[i/100.0 for i in range(75,90,5)]
# }
# param_test6 = {
#  'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
# }
# param_test7 = {
#  'reg_alpha':[0, 0.001, 0.005, 0.01, 0.05]
# }