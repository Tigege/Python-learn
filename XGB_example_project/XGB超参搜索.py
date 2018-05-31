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

#alg����XGBOOST,X_train����ѵ������������Ϣ��Y_train����ѵ�����ݱ�ǩ��Ϣ  X_testdata���ҪԤ���ֵ
def XGBmodelfit(alg, X_train, Y_train,X_test=None,Y_test=None,X_predictions=None,useTrainCV=True, cv_folds=5, early_stopping_rounds=200):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(X_train, label=Y_train)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='auc', early_stopping_rounds=early_stopping_rounds, show_stdv=False)
        alg.set_params(n_estimators=cvresult.shape[0])

    #ѵ��ģ��
    alg.fit(X_train, Y_train,eval_metric='auc')

    #Ԥ����:
    dtrain_predictions = alg.predict(X_test)  #��� 0 �� 1
    # dtrain_predprob = alg.predict_proba(X_test)[:,1]   #�������

    #��ӡ������Ϣ:
    print("\nModel Report")
    print("Accuracy  (Train) : %.4g" % metrics.accuracy_score(Y_test, dtrain_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(Y_test, dtrain_predictions))
    print(alg)
    print("the best:")
    print(cvresult.shape[0])
    plot_importance(alg)
    plt.show()

    # feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    # feat_imp.plot(kind='bar', title='Feature Importances')
    # plt.ylabel('Feature Importance Score')

dataset_X,dataset_Y=return_data.return_tarin_data()

X_train, X_test, y_train, y_test = train_test_split(dataset_X, dataset_Y,
                                                  test_size=0.2,
                                                  random_state=45)

xgb1 = XGBClassifier(
     learning_rate =0.1,
     n_estimators=1000,
     max_depth=5,
     min_child_weight=1,
     gamma=0,
     subsample=0.8,
     colsample_bytree=0.8,
     objective= 'binary:logistic',
     nthread=4,
     scale_pos_weight=1,
     seed=27)

# XGBmodelfit(xgb1,X_train,y_train,X_test,y_test)


# gsearch1 = GridSearchCV(estimator = XGBClassifier(
#        learning_rate =0.1, n_estimators=140, max_depth=9,
#        min_child_weight=1, gamma=0, subsample=0.8,colsample_bytree=0.8,
#        objective= 'binary:logistic', nthread=4,scale_pos_weight=1, seed=27),
#        param_grid=param_grid,cv=5)
param_grid = {
 'max_depth':range(3,7,2),
 'min_child_weight':range(1,6,2)
}
param_test2 = {
 'max_depth':[4,5,6],
 'min_child_weight':[4,5,6]
}
param_test2b = {
 'min_child_weight':[6,8,10,12]
 }
#[0.0, 0.1, 0.2, 0.3, 0.4]
param_test3 = {
 'gamma':[i/10.0 for i in range(0,5)]
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
gsearch1 = GridSearchCV(estimator =
                        XGBClassifier(
                         learning_rate =0.1,
                         n_estimators=484,
                         max_depth=5,
                         min_child_weight=1,
                         gamma=0,
                         subsample=0.8,
                         colsample_bytree=0.8,
                         objective= 'binary:logistic',
                         nthread=4,
                         scale_pos_weight=1,
                         seed=27),
         param_grid=param_test3,cv=5)

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