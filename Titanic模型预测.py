# -*- coding: UTF-8 -*-
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.feature_selection import SelectKBest
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
import warnings
from sklearn import cross_validation


train = pd.read_csv("train1.csv")
test = pd.read_csv("test1.csv")
PassengerId=test['PassengerId']
#print train.info()
all_data = pd.concat([train, test])
#print all_data.info()
all_data = all_data[['Survived', 'Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Title', 'FamilyLabel', 'Deck', 'TicketGroup']]
#print all_data.info()
all_data = pd.get_dummies(all_data)

train = all_data[all_data['Survived'].notnull()]
test = all_data[all_data['Survived'].isnull()].drop('Survived', axis = 1)
X = train.as_matrix()[:, 1:]
y = train.as_matrix()[:, 0]
#print type(X)

#利用随机森林进行预测
# pipe = Pipeline([('select', SelectKBest(k = 20)),
#                  ('classify', RandomForestClassifier(random_state = 10, max_features = 'sqrt'))])
# param_test = {'classify__n_estimators': list(range(20, 50, 2)),
#               'clsssify__max_depth': list(range(3, 60, 3))}
# gsearch = GridSearchCV(estimator = pipe, param_grid = param_test,scoring = 'roc_auc', cv = 10)
# gsearch.fit(X, y)
# pipe=Pipeline([('select',SelectKBest(k=20)), 
#                ('classify', RandomForestClassifier(random_state = 10, max_features = 'sqrt'))])
 
# param_test = {'classify__n_estimators':list(range(20,50,2)), 
#               'classify__max_depth':list(range(3,60,3))}
# gsearch = GridSearchCV(estimator = pipe, param_grid = param_test, scoring='roc_auc', cv=10)
# gsearch.fit(X,y)
# print(gsearch.best_params_, gsearch.best_score_)
select = SelectKBest(k = 20)
clf = RandomForestClassifier(random_state = 10, warm_start = True, 
                                  n_estimators = 24,
                                  max_depth = 6, 
                                  max_features = 'sqrt')
pipeline = make_pipeline(select, clf)
pipeline.fit(X, y)
predictions = pipeline.predict(test)

submission = pd.DataFrame({"PassengerId": PassengerId, "Survived": predictions.astype(np.int32)})
submission.to_csv("submission.csv", index=False)





