#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 17:59:19 2021

@author: downey
"""

#------------------------------- Packages ---------------------------------
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
import os
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.metrics import plot_confusion_matrix

from imblearn.over_sampling import SMOTE
from matplotlib import pyplot

from pprint import pprint

#------------------------------- Import Data ---------------------------------
path = '/Users/downey/Desktop/passage.csv'
data = np.loadtxt(path, dtype=float, delimiter=',', skiprows = 1)
d1 = pd.read_csv("/Users/downey/Desktop/passage.csv")
data.shape[1]

#Get columns Name
columns = d1.columns.tolist()
columns.remove("Healthy")

#Split X and Y data
x,y = np.split(data, (data.shape[1] - 1,), axis = 1)

#------------------------------ MinMaxScaler ---------------------------------
#MINMAX SCaler
min_max_scaler = preprocessing.MinMaxScaler()
x = min_max_scaler.fit_transform(x)

data = np.c_[x, y]


#------------------------------ SMOTE ----------------------------------------
# balance classes
smo = SMOTE(random_state = 25)
x_smo, y_smo = smo.fit_resample(x, y)
data = np.c_[x_smo, y_smo]

#x_smo Name
x_smo = pd.DataFrame(x_smo)
x_smo.columns = columns
#y_smo Name
y_smo = pd.DataFrame(y_smo)
y_smo.columns = ["Healthy"]


#-------------------------- Train/Testing Split ------------------------------
xtrain, xtest, ytrain, ytest = train_test_split(x_smo, y_smo, test_size=0.30, random_state=15)
print(Counter(ytest['Healthy']))
#Disease(0) - 19
#Healthy(1) - 7
ytrain = np.array(ytrain)

#------------------------------------ Default LR ------------------------------
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(random_state = 25)
lr.fit(xtrain, ytrain)
pred_lr = lr.predict(xtest)

print(accuracy_score(pred_lr,ytest))
print(confusion_matrix(ytest,pred_lr))
print(classification_report(ytest,pred_lr))

rfc_cv_score = cross_val_score(lr, x_smo, y_smo, cv=10, scoring="roc_auc")
print(np.average(rfc_cv_score))


















