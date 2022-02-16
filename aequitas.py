#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 16:49:14 2021

@author: downey
"""

from sklearn import svm
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pandas as pd
import matplotlib.pyplot as plt

import os
import os.path
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
from aequitas.group import Group
from aequitas.bias import Bias
from aequitas.fairness import Fairness
from aequitas.plotting import Plot
from aequitas.preprocessing import preprocess_input_df
from collections import Counter
from sklearn.metrics import accuracy_score

from imblearn.over_sampling import SMOTE


path = '/Users/downey/Desktop/passage.csv'
data = np.loadtxt(path, dtype=float, delimiter=',', skiprows = 1)

x,y = np.split(data, (data.shape[1] - 1,), axis=1)
df = pd.DataFrame(data = x)
df = pd.get_dummies(df, columns = [0,1,2,3,4,5], drop_first = True)

x = np.array(df)
min_max_scaler = preprocessing.MinMaxScaler()
x = min_max_scaler.fit_transform(x)

data = np.c_[x, y]

#SMOTE
# smo = SMOTE(random_state=45)
# x_smo, y_smo = smo.fit_resample(x, y)
# data = np.c_[x_smo, y_smo]


# x_smo = np.delete(x_smo, 55, 1)
# x_smo = np.delete(x_smo, 55, 1)


#----------------------------------------------------
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.30, random_state = 10)
clf = svm.SVC(C=128, kernel='rbf', gamma=1.0, decision_function_shape='ovr')
clf.fit(xtrain, ytrain.ravel())
score = clf.predict(xtest)
print(accuracy_score(score,ytest))




#------Score column
score = pd.DataFrame(score)
score.columns = ["score"]


#------Entity_id
entity_id = {'entity_id':[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]}
entity_id = pd.DataFrame(entity_id)

#------Label Value
data = pd.read_csv('/Users/downey/Desktop/v1_passage.csv')
x,y = np.split(data, (data.shape[1] - 1,), axis=1)
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.30, random_state=10)
xtest = xtest.iloc[:,0:4]


xtest = xtest.reset_index(drop=True)
ytest = ytest.reset_index(drop=True)
ytest.columns = ["label_value"]

#------The Dataframe
#data = pd.concat([entity_id, score, ytest, xtest], axis=1)
data = pd.concat([score, ytest, xtest], axis=1)

#----------------------------------- Aequitas ---------------------------------

aq_palette = sns.diverging_palette(225, 35, n=2)
by_sex = sns.countplot(x="Gender", hue="score", data=data, palette=aq_palette)
by_sex = sns.countplot(x="age_group", hue="score", data=data, palette=aq_palette)



df, _ = preprocess_input_df(data)

g = Group()
xtab, _ = g.get_crosstabs(df)

absolute_metrics = g.list_absolute_metrics(xtab)
print(xtab[[col for col in xtab.columns if col not in absolute_metrics]])

zz = xtab[['attribute_name', 'attribute_value'] + absolute_metrics].round(2)
print(zz)


aqp = Plot()
fnr = aqp.plot_group_metric(xtab, 'fnr')

fnr = aqp.plot_group_metric(xtab, 'fnr', min_group_size=0.05)

