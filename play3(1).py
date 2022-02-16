from sklearn import svm
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
from sklearn.model_selection import train_test_split
import math
from sklearn.model_selection import KFold,LeaveOneOut,LeavePOut,ShuffleSplit
from sklearn import preprocessing
import pandas as pd
import matplotlib.pyplot as plt
import csv
import string
from numba import cuda
import os
import os.path
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
import seaborn as sns
from aequitas.group import Group
from aequitas.bias import Bias
from aequitas.fairness import Fairness
from aequitas.plotting import Plot
from aequitas.preprocessing import preprocess_input_df

from imblearn.over_sampling import SMOTE


path = 'final_V.csv'
data = np.loadtxt(path, dtype=float, delimiter=',', skiprows=1)

x,y = np.split(data, (data.shape[1] - 1,), axis=1)
df = pd.DataFrame(data=x)
df = pd.get_dummies(df, columns = [0,1,2,3,4,5], drop_first=True)

x = np.array(df)
min_max_scaler = preprocessing.MinMaxScaler()
x = min_max_scaler.fit_transform(x)

data = np.c_[x, y]



# validate feature importance
model = RandomForestRegressor()
x,y = np.split(data, (data.shape[1] - 1,), axis=1)
model.fit(x, y.ravel())
importance = model.feature_importances_

# balance classes
smo = SMOTE(random_state=42)
x_smo, y_smo = smo.fit_resample(x, y)
data = np.c_[x_smo, y_smo]


# 5-fold cross-validation
score_svm = 0
score_rf = 0
score_lr = 0
score_nb = 0
score_gbdt = 0
kf = KFold(n_splits=10, shuffle=True, random_state=1)
for train, test in kf.split(data):
    train = data[train]
    test = data[test]
    xtrain, ytrain = np.split(train, (data.shape[1] - 1,), axis=1)
    xtest, ytest = np.split(test, (data.shape[1] - 1,), axis=1)


    clf = svm.SVC(C=128, kernel='rbf', gamma=0.0001, decision_function_shape='ovr')
    clf.fit(xtrain, ytrain.ravel())
    ypredict = clf.predict(xtest)
    score_svm += clf.score(xtest, ytest)

    rfc = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2, bootstrap=True)
    rfc = rfc.fit(xtrain, ytrain.ravel())
    score_rf += rfc.score(xtest, ytest)

    clf = LogisticRegression(penalty='l2', C=1.0)
    clf = clf.fit(xtrain, ytrain.ravel())
    score_lr += clf.score(xtest, ytest)

    gnb = GaussianNB()
    gnb = gnb.fit(xtrain, ytrain.ravel())
    score_nb += gnb.score(xtest, ytest)

    model = GradientBoostingClassifier()
    model = model.fit(xtrain, ytrain.ravel())
    score_gbdt += model.score(xtest, ytest)

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.30, random_state=10)
clf = svm.SVC(C=64, kernel='rbf', gamma=0.0001, decision_function_shape='ovr')
clf.fit(xtrain, ytrain.ravel())
score = clf.predict(xtest)
score = pd.DataFrame(score)
score.columns = ["score"]
print(confusion_matrix(ytest, score))

entity_id = {'entity_id':[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]}
entity_id = pd.DataFrame(entity_id)

data = pd.read_csv('v1.csv', sep=',')
x,y = np.split(data, (data.shape[1] - 1,), axis=1)
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.30, random_state=10)
xtest = xtest.iloc[:,0:4]

xtest = xtest.reset_index(drop=True)
ytest = ytest.reset_index(drop=True)
ytest.columns = ["label_value"]

data = pd.concat([entity_id, score, ytest, xtest], axis=1)
df, _ = preprocess_input_df(data)

g = Group()
xtab, _ = g.get_crosstabs(df)

absolute_metrics = g.list_absolute_metrics(xtab)
print(xtab[[col for col in xtab.columns if col not in absolute_metrics]])

zz = xtab[['attribute_name', 'attribute_value'] + absolute_metrics].round(2)
print()