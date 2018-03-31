---
layout:     post
title:      Pipeline Processing with Hyper-Parameter-Tuning in Python
subtitle:   Demo in Titanic Data
date:       2017-08-07
author:     Haby
header-img: img/post-bg-2015.jpg
catalog: true
tags:
    - Python
    - My Work
    - Machine Learning
    - Pipeline
    - Algorithms
---

```python
import warnings
warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import sem

from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cross_validation import cross_val_score,KFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.grid_search import GridSearchCV
import xgboost as xgb
```


```python
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

print(train.info())
print("-"*40)
print(test.info())

# NAs in Age and Cabin for both datasets and 2 NAs in Embarked in train and 1 NA in Fare in test
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 12 columns):
    PassengerId    891 non-null int64
    Survived       891 non-null int64
    Pclass         891 non-null int64
    Name           891 non-null object
    Sex            891 non-null object
    Age            714 non-null float64
    SibSp          891 non-null int64
    Parch          891 non-null int64
    Ticket         891 non-null object
    Fare           891 non-null float64
    Cabin          204 non-null object
    Embarked       889 non-null object
    dtypes: float64(2), int64(5), object(5)
    memory usage: 83.6+ KB
    None
    ----------------------------------------
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 418 entries, 0 to 417
    Data columns (total 11 columns):
    PassengerId    418 non-null int64
    Pclass         418 non-null int64
    Name           418 non-null object
    Sex            418 non-null object
    Age            332 non-null float64
    SibSp          418 non-null int64
    Parch          418 non-null int64
    Ticket         418 non-null object
    Fare           417 non-null float64
    Cabin          91 non-null object
    Embarked       418 non-null object
    dtypes: float64(2), int64(4), object(5)
    memory usage: 36.0+ KB
    None



```python
# Exploratory data analysis

for d in [train,test] :
    # convert pclass from int to str
    d['Pclass'] = d['Pclass'].astype('str')

    # split title from name
    d['Title'] = d['Name'].apply(lambda x:x.split(',')[1].split('.')[0].strip())

    # re-group into 5 parts
    d['Title'].replace(['Capt', 'Col', 'Major', 'Dr', 'Rev'] ,'Officer',inplace = True)
    d['Title'].replace(['Don', 'Sir', 'the Countess', 'Dona', 'Lady']  ,'Royalty',inplace = True)
    d['Title'].replace( ['Mme', 'Ms', 'Mrs'] ,'Mrs',inplace = True)
    d['Title'].replace(['Mlle', 'Miss'] ,'Miss',inplace = True)
    d['Title'].replace(['Master','Jonkheer'] ,'Master',inplace = True)

    # imputation
    # Fill Cabin with X
    d['Cabin'].fillna('X',inplace = True)

    # Pick up first letter of each Cabin as Deck
    d['Deck'] = d['Cabin'].str.get(0)

    # Fill Embarked with C
    d['Embarked'].fillna('C',inplace = True)

    # Fill Age with mean of grouped Pclass, Sex, New_title
    d['Age'] = d.groupby(['Pclass','Sex','Title'])['Age'].transform(lambda x : x.fillna(x.mean()))

    # Fill Fare with mean of same Pclass and Cabin
    d["Fare"] = d.groupby(["Pclass","Embarked","Cabin"])['Fare'].transform(lambda x: x.fillna(x.mean()))

    # Family size
    d["Family_Size"] = d["SibSp"] + d['Parch'] + 1

    # The number of ppl who has the same ticket number
    grouped_ticket = dict(d['Ticket'].value_counts())
    d['grouped_ticket'] = d['Ticket'].apply(lambda x:grouped_ticket[x])

    # familysize / groupticket to str
    d['Family_Size'] = d['Family_Size'].astype('str')
    d['grouped_ticket'] = d['grouped_ticket'].astype('str')

    # drop some columns
    d.drop(["PassengerId","Name","SibSp","Parch","Ticket","Cabin"],axis = 1,inplace = True)

```


```python
# data proprecessing

# on-hot-encoding to dummy variables
df = pd.get_dummies(train)

# seperate into data and target
X = df.drop(['Survived'],axis = 1)
y = df['Survived']

# train and test split into two parts
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = .3,random_state = 13)
```


```python
# function estimator mean score for each cv results
def evaluate_cross_validation(clf, X, y, K):
    """
    A function that calculate mean of cross validation score and mean square error.
    Input : classfication method(algorithm model), train(X),test(y),number of folder(k)
    """
    # KFold parameters with shuffle = True
    cv = KFold(len(y), K, shuffle=True, random_state = 13)
    # default cv score is accurary, can be changed to rmse and etc.
    scores = cross_val_score(clf, X, y, cv=cv)
    # print scores for each cv results and mean results
    print(scores)
    print('Mean score: %.3f (+/-%.3f)' % (scores.mean(), sem(scores)))
```


```python
# pipeline for different model
clf_1 = Pipeline([('scalar',StandardScaler()),('feature_selectoin', SelectKBest(k = 22)),('rf',RandomForestClassifier(random_state = 13))])
clf_2 = Pipeline([('scalar',StandardScaler()),('feature_selectoin', SelectKBest(k = 25)),('dtc',DecisionTreeClassifier(random_state = 13))])
clf_3 = Pipeline([('scalar',StandardScaler()),('feature_selectoin', SelectKBest(k = 30)),('gbm',GradientBoostingClassifier(random_state = 13))])
clf_4 = Pipeline([('scalar',StandardScaler()),('feature_selectoin', SelectKBest(k = 20)),('svm',SVC(random_state = 13))])
clf_5 = Pipeline([('scalar',StandardScaler()),('feature_selectoin', SelectKBest(k = 25)),('xtc',ExtraTreesClassifier(random_state = 13))])
clf_6 = Pipeline([('scalar',StandardScaler()),('feature_selectoin', SelectKBest(k = 20)),('neural',MLPClassifier(random_state = 13))])
clf_7 = Pipeline([('scalar',StandardScaler()),('feature_selectoin', SelectKBest(k = 40)),('xgb',xgb.XGBClassifier(random_state = 13))])


for clf in [clf_1,clf_2,clf_3,clf_4,clf_5,clf_6,clf_7] :
    print(evaluate_cross_validation(clf,x_train,y_train,5))

# regular result
#Mean score: 0.804 (+/-0.007)
#Mean score: 0.770 (+/-0.016)
#Mean score: 0.828 (+/-0.010)
#Mean score: 0.828 (+/-0.011)
#Mean score: 0.801 (+/-0.015)
#Mean score: 0.811 (+/-0.012)
#Mean score: 0.825 (+/-0.012)


# feature selection result
#Mean score: 0.807 (+/-0.010)
#Mean score: 0.799 (+/-0.018)
#Mean score: 0.831 (+/-0.020)
#Mean score: 0.830 (+/-0.014)
#Mean score: 0.806 (+/-0.009)
#Mean score: 0.823 (+/-0.014)
#Mean score: 0.825 (+/-0.012)

```

    [0.832      0.8        0.792      0.78225806 0.83064516]
    Mean score: 0.807 (+/-0.010)
    None
    [0.8        0.84       0.784      0.74193548 0.83064516]
    Mean score: 0.799 (+/-0.018)
    None
    [0.84       0.856      0.776      0.7983871  0.88709677]
    Mean score: 0.831 (+/-0.020)
    None
    [0.856      0.84       0.776      0.83870968 0.83870968]
    Mean score: 0.830 (+/-0.014)
    None
    [0.824      0.824      0.776      0.7983871  0.80645161]
    Mean score: 0.806 (+/-0.009)
    None
    [0.84       0.832      0.768      0.83870968 0.83870968]
    Mean score: 0.823 (+/-0.014)
    None  
    [0.824      0.856      0.8        0.7983871  0.84677419]
    Mean score: 0.825 (+/-0.012)
    None



```python
# Hyper parameters tuning for rf model
parameters = {'rf__n_estimators' : np.arange(320,340,2),
              'rf__max_depth':np.arange(5,7,1),
              'rf__min_samples_split':np.arange(2,3,1),
              'rf__min_samples_leaf':np.arange(1,2,1)}
GridS = GridSearchCV(clf_1,parameters, verbose = 1,cv = 10,n_jobs = -1)
GridS.fit(x_train,y_train)
print('-'*40)
print('Inner test score : %.5f' %GridS.best_score_ )
print('Best Parameter : %s'%GridS.best_params_)
print('-'*40)
print("Outside test score : %.5f" %GridS.score(x_test,y_test))

#----------------------------------------
#Inner test score : 0.83628
#Best Parameter : {'rf__max_depth': 6, 'rf__min_samples_leaf': 1, 'rf__min_samples_split': 2, 'rf__n_estimators': 328}
#----------------------------------------
#Outside test score : 0.81716
```

    Fitting 10 folds for each of 20 candidates, totalling 200 fits


    [Parallel(n_jobs=-1)]: Done 136 tasks      | elapsed:   15.3s
    [Parallel(n_jobs=-1)]: Done 200 out of 200 | elapsed:   20.0s finished


    ----------------------------------------
    Inner test score : 0.83628
    Best Parameter : {'rf__max_depth': 6, 'rf__min_samples_leaf': 1, 'rf__min_samples_split': 2, 'rf__n_estimators': 328}
    ----------------------------------------
    Outside test score : 0.81716



```python
# Hyper parameters tuning for dtc model
parameters = {'dtc__max_features' : [None],
              'dtc__max_depth':np.arange(2,6,1),
              'dtc__min_samples_split':np.arange(2,3,1),
              'dtc__min_samples_leaf':np.arange(1,4,1)}
GridS = GridSearchCV(clf_2,parameters, verbose = 1,cv = 10,n_jobs = -1)
GridS.fit(x_train,y_train)
print('-'*40)
print('Inner test score : %.5f' %GridS.best_score_ )
print('Best Parameter : %s'%GridS.best_params_)
print('-'*40)
print("Outside test score : %.5f" %GridS.score(x_test,y_test))

#----------------------------------------
#Inner test score : 0.82504
#Best Parameter : {'dtc__max_depth': 4, 'dtc__max_features': None, 'dtc__min_samples_leaf': 1, 'dtc__min_samples_split': 2}
#----------------------------------------
#Outside test score : 0.81716
```

    Fitting 10 folds for each of 12 candidates, totalling 120 fits
    ----------------------------------------
    Inner test score : 0.82504
    Best Parameter : {'dtc__max_depth': 4, 'dtc__max_features': None, 'dtc__min_samples_leaf': 1, 'dtc__min_samples_split': 2}
    ----------------------------------------
    Outside test score : 0.81716


    [Parallel(n_jobs=-1)]: Done 120 out of 120 | elapsed:    0.2s finished



```python
# Hyper parameters tuning for dtc model
parameters = {'gbm__n_estimators' : np.arange(606,618,2),
              'gbm__learning_rate':[.2],
              'gbm__min_samples_split':np.arange(2,3,1),
              'gbm__min_samples_leaf':np.arange(1,2,1),
              'gbm__subsample' : [.6],
              'gbm__max_depth':np.arange(3,4,1),
              'gbm__criterion':['mae']}
GridS = GridSearchCV(clf_3,parameters, verbose = 1,cv = 10,n_jobs = -1)
GridS.fit(x_train,y_train)
print('-'*40)
print('Inner test score : %.5f' %GridS.best_score_ )
print('Best Parameter : %s'%GridS.best_params_)
print('-'*40)
print("Outside test score : %.5f" %GridS.score(x_test,y_test))
#----------------------------------------
#Inner test score : 0.81701
#Best Parameter : {'gbm__criterion': 'mae', 'gbm__learning_rate': 0.2, 'gbm__max_depth': 3, 'gbm__min_samples_leaf': 1, 'gbm__min_samples_split': 2, 'gbm__n_estimators': 610, 'gbm__subsample': 0.6}
#----------------------------------------
#Outside test score : 0.82836
```

    Fitting 10 folds for each of 6 candidates, totalling 60 fits


    [Parallel(n_jobs=-1)]: Done  58 out of  60 | elapsed:  1.3min remaining:    2.8s
    [Parallel(n_jobs=-1)]: Done  60 out of  60 | elapsed:  1.3min finished


    ----------------------------------------
    Inner test score : 0.81701
    Best Parameter : {'gbm__criterion': 'mae', 'gbm__learning_rate': 0.2, 'gbm__max_depth': 3, 'gbm__min_samples_leaf': 1, 'gbm__min_samples_split': 2, 'gbm__n_estimators': 610, 'gbm__subsample': 0.6}
    ----------------------------------------
    Outside test score : 0.82836



```python
# Hyper parameters tuning for svm model
parameters = {'svm__C' : np.arange(1,5,1),
              'svm__gamma':[.03,.025],
              'svm__kernel':['rbf'],
              'svm__tol':[0.0001,.001]}
GridS = GridSearchCV(clf_4,parameters, verbose = 1,cv = 10,n_jobs = -1)
GridS.fit(x_train,y_train)
print('-'*40)
print('Inner test score : %.5f' %GridS.best_score_ )
print('Best Parameter : %s'%GridS.best_params_)
print('-'*40)
print("Outside test score : %.5f" %GridS.score(x_test,y_test))

#----------------------------------------
#Inner test score : 0.82825
#Best Parameter : {'svm__C': 1, 'svm__gamma': 0.03, 'svm__kernel': 'rbf', 'svm__tol': 0.0001}
#----------------------------------------
#Outside test score : 0.80970
```

    Fitting 10 folds for each of 16 candidates, totalling 160 fits
    ----------------------------------------
    Inner test score : 0.82825
    Best Parameter : {'svm__C': 1, 'svm__gamma': 0.03, 'svm__kernel': 'rbf', 'svm__tol': 0.0001}
    ----------------------------------------
    Outside test score : 0.80970


    [Parallel(n_jobs=-1)]: Done 160 out of 160 | elapsed:    0.8s finished



```python
# Hyper parameters tuning for xtc model
parameters = {'xtc__max_features' : [None],
              'xtc__n_estimators' : np.arange(4,20,2),
              'xtc__max_depth':np.arange(3,5,1),
              'xtc__min_samples_split':np.arange(2,3,1),
              'xtc__min_samples_leaf':np.arange(1,3,1)}
GridS = GridSearchCV(clf_5,parameters, verbose = 1,cv = 10,n_jobs = -1)
GridS.fit(x_train,y_train)
print('-'*40)
print('Inner test score : %.5f' %GridS.best_score_ )
print('Best Parameter : %s'%GridS.best_params_)
print('-'*40)
print("Outside test score : %.5f" %GridS.score(x_test,y_test))

#----------------------------------------
#Inner test score : 0.83146
#Best Parameter : {'xtc__max_depth': 4, 'xtc__max_features': None, 'xtc__min_samples_leaf': 2, 'xtc__min_samples_split': 2, 'xtc__n_estimators': 8}
#----------------------------------------
#Outside test score : 0.80970
```

    Fitting 10 folds for each of 32 candidates, totalling 320 fits


    [Parallel(n_jobs=-1)]: Done  17 tasks      | elapsed:    0.5s


    ----------------------------------------
    Inner test score : 0.83146
    Best Parameter : {'xtc__max_depth': 4, 'xtc__max_features': None, 'xtc__min_samples_leaf': 2, 'xtc__min_samples_split': 2, 'xtc__n_estimators': 8}
    ----------------------------------------
    Outside test score : 0.80970


    [Parallel(n_jobs=-1)]: Done 320 out of 320 | elapsed:    1.7s finished



```python
# Hyper parameters tuning for mpl model
parameters = {'neural__alpha' : [.0001],
              'neural__max_iter' : np.arange(100,150,5),
              'neural__tol':[.00001],
              'neural__solver':['adam'],
              'neural__activation':['relu']}
GridS = GridSearchCV(clf_6,parameters, verbose = 1,cv = 10,n_jobs = -1)
GridS.fit(x_train,y_train)
print('-'*40)
print('Inner test score : %.5f' %GridS.best_score_ )
print('Best Parameter : %s'%GridS.best_params_)
print('-'*40)
print("Outside test score : %.5f" %GridS.score(x_test,y_test))

#----------------------------------------
#Inner test score : 0.81862
#Best Parameter : {'neural__activation': 'relu', 'neural__alpha': 0.0001, 'neural__max_iter': 135, 'neural__solver': 'adam', 'neural__tol': 1e-05}
#----------------------------------------
#Outside test score : 0.80597
```

    Fitting 10 folds for each of 10 candidates, totalling 100 fits


    [Parallel(n_jobs=-1)]: Done 100 out of 100 | elapsed:    7.1s finished


    ----------------------------------------
    Inner test score : 0.81862
    Best Parameter : {'neural__activation': 'relu', 'neural__alpha': 0.0001, 'neural__max_iter': 135, 'neural__solver': 'adam', 'neural__tol': 1e-05}
    ----------------------------------------
    Outside test score : 0.80597



```python
xgb.XGBClassifier().get_params()
# Hyper parameters tuning for mpl model
parameters = {'xgb__learning_rate' : [.002],
              'xgb__max_depth' : np.arange(3,4,1),
              #'xgb__colsample_bylevel':np.arange(1,4,2),
             # 'xgb__colsample_bytree':np.arange(1,4,2),
              'xgb__gamma':[0.3],
              'xgb__n_estimators':np.arange(200,300,5),
              'xgb__subsample':[.8]}
GridS = GridSearchCV(clf_7,parameters, verbose = 1,cv = 10,n_jobs = -1)
GridS.fit(x_train,y_train)
print('-'*40)
print('Inner test score : %.5f' %GridS.best_score_ )
print('Best Parameter : %s'%GridS.best_params_)
print('-'*40)
print("Outside test score : %.5f" %GridS.score(x_test,y_test))
print('-'*40)
print("Inner Test Report:\n",classification_report(GridS.predict(x_train),y_train))
print('-'*40)
print("Predict Report:\n",classification_report(GridS.predict(x_test),y_test))

#----------------------------------------
#Inner test score : 0.82665
#Best Parameter : {'xgb__gamma': 0.3, 'xgb__learning_rate': 0.002, 'xgb__max_depth': 3, 'xgb__n_estimators': 220, 'xgb__subsample': 0.8}
#----------------------------------------
#Outside test score : 0.82463
```

    Fitting 10 folds for each of 20 candidates, totalling 200 fits

    [Parallel(n_jobs=-1)]: Done 200 out of 200 | elapsed:    8.7s finished


    ----------------------------------------
    Inner test score : 0.82665
    Best Parameter : {'xgb__gamma': 0.3, 'xgb__learning_rate': 0.002, 'xgb__max_depth': 3, 'xgb__n_estimators': 220, 'xgb__subsample': 0.8}
    ----------------------------------------
    Outside test score : 0.82463
    ----------------------------------------
    Inner Test Report:
                  precision    recall  f1-score   support

              0       0.90      0.85      0.88       401
              1       0.76      0.84      0.80       222

    avg / total       0.85      0.85      0.85       623

    ----------------------------------------
    Predict Report:
                  precision    recall  f1-score   support

              0       0.86      0.86      0.86       170
              1       0.76      0.76      0.76        98

    avg / total       0.82      0.82      0.82       268




```python
# best single model : xgboost model with outside test score : 0.82463
# and from classfication report I find that model has a bad result in predicting who is survived but has a better result in who is died.
```
