
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
model.fit(train_x, train_y)

from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB(alpha=0.01)
model.fit(train_x, train_y)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(penalty='l2')
model.fit(train_x, train_y)

from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier(n_estimators=200)
model.fit(train_x, train_y)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=8)
model.fit(train_x, train_y)

from sklearn.svm import SVC
model = SVC(kernel='rbf', probability=True)
model.fit(train_x, train_y)


from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
model = SVC(kernel='rbf', probability=True)
param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}
grid_search = GridSearchCV(model, param_grid, n_jobs=1, verbose=1)
grid_search.fit(train_x, train_y)
best_parameters = grid_search.best_estimator_.get_params()
for para, val in list(best_parameters.items()):
    print(para, val)
model = SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)
model.fit(train_x, train_y)