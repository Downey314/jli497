#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 22:20:37 2021

@author: downey
"""

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
from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.metrics import plot_confusion_matrix

from imblearn.over_sampling import SMOTE
from matplotlib import pyplot



path = '/Users/downey/Desktop/passage.csv'
data = np.loadtxt(path, dtype=float, delimiter=',', skiprows = 1)
d1 = pd.read_csv("/Users/downey/Desktop/passage.csv")
data.shape[1]

#Get Columns name
columns = d1.columns.tolist()
columns.remove("Healthy")
columns.remove("age_group")
columns.remove("Gender")

columns.append("Age_2")
columns.append("Age_3")
columns.append("Male")




#Dummy Variables
x,y = np.split(data, (data.shape[1] - 1,), axis = 1)
df = pd.DataFrame(data = x)
df = pd.get_dummies(df, columns = [0,1], drop_first = True)

#MINMAX SCaler
x = np.array(df)
min_max_scaler = preprocessing.MinMaxScaler()
x = min_max_scaler.fit_transform(x)

data = np.c_[x, y]
#data = pd.DataFrame(data)


# balance classes
smo = SMOTE(random_state = 45)
x_smo, y_smo = smo.fit_resample(x, y)
data = np.c_[x_smo, y_smo]

#x_smo Name
x_smo = pd.DataFrame(x_smo)
x_smo.columns = columns
#y_smo Name
y_smo = pd.DataFrame(y_smo)
y_smo.columns = ["Healthy"]


x_smo.loc[x_smo.Age_2 < 0.5, "Age_2"] = 0
x_smo.loc[x_smo.Age_2 > 0.5, "Age_2"] = 1

x_smo.loc[x_smo.Age_3 < 0.5, "Age_3"] = 0
x_smo.loc[x_smo.Age_3 > 0.5, "Age_3"] = 1

x_smo.loc[x_smo.Male < 0.5, "Male"] = 0
x_smo.loc[x_smo.Male > 0.5, "Male"] = 1


xtrain, xtest, ytrain, ytest = train_test_split(x_smo, y_smo, test_size=0.30, random_state=10)


#RF ALGORTIHM
score_rf = 0

kf = KFold(n_splits=10, shuffle=True, random_state = 1)
for train, test in kf.split(data):
    train = data[train]
    test = data[test]
    xtrain, ytrain = np.split(train, (data.shape[1] - 1,), axis=1)
    xtest, ytest = np.split(test, (data.shape[1] - 1,), axis=1)

    #rfc = RandomForestClassifier(n_estimators=250, max_depth=None, min_samples_split=2, bootstrap=True)
    rfc = RandomForestClassifier()
    rfc = rfc.fit(xtrain, ytrain.ravel())
    #print(rfc.score(xtest,ytest))
    score_rf += rfc.score(xtest, ytest)



time_start=time.time()

model_RF = RandomForestClassifier()
model_RF.fit(xtrain, ytrain)   
pred_rf = model_RF.predict(xtest)

time_end=time.time()
print('time cost',time_end-time_start,'s')


print(accuracy_score(pred_rf,ytest))
print(confusion_matrix(ytest,pred_rf))
print(classification_report(ytest,pred_rf))

X_train_prediction = model_RF.predict(xtrain)
training_data_accuracy = accuracy_score(ytrain, X_train_prediction)
print("Accuracy Score of training data: ", training_data_accuracy)

#accuracy score on training data
X_test_prediction = model_RF.predict(xtest)
testing_data_accuracy = accuracy_score(ytest, X_test_prediction)
print("Accuracy Score of test data: ", testing_data_accuracy)


import itertools
def plot_confusion_matrix(cm, classes, normalize = False,
                          title='Confusion matrix',
                          cmap=plt.cm.Greens): # can change color 
    plt.figure(figsize = (10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, size = 24)
    plt.colorbar(aspect=4)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, size = 14)
    plt.yticks(tick_marks, classes, size = 14)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    # Label the plot
    for i, j in itertools.product(range(cm.shape[0]),   range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), 
                 fontsize = 20,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
        plt.grid(None)
        plt.tight_layout()
        plt.ylabel('True label', size = 18)
        plt.xlabel('Predicted label', size = 18)

# Let's plot it out
cm = confusion_matrix(ytest, pred_rf)
plot_confusion_matrix(cm, classes = ['0 - Healthy','1 - Disease'],
                      title = 'Healthy_status Confusion Matrix')



from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
model = RandomForestClassifier()
n_estimators = [10, 25, 50, 100, 150,200, 250, 500]
max_features = ['sqrt', 'log2']
max_depth=[None, 5, 10]
criterion=["gini", "entropy"]
# define grid search
grid = dict(n_estimators=n_estimators,max_features=max_features,max_depth=max_depth)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(x_smo, y_smo)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
    
    
model_LR = RandomForestClassifier(n_estimators=500, max_depth=10, min_samples_split=2, bootstrap=True)
model_LR.fit(xtrain, ytrain)
pred_lr = model_LR.predict(xtest)
print(accuracy_score(pred_lr,ytest))
print(confusion_matrix(ytest,pred_lr))
print(classification_report(ytest,pred_lr))

cm = confusion_matrix(ytest, pred_lr)
plot_confusion_matrix(cm, classes = ['0 - Healthy','1 - Disease'],
                      title = 'Healthy_status Confusion Matrix')



from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve, f1_score
train_probs = model_LR.predict_proba(xtrain)[:,1] 
probs = model_LR.predict_proba(xtest)[:, 1]
train_predictions = model_LR.predict(xtrain)

def evaluate_model(y_pred, probs,train_predictions, train_probs):
    baseline = {}
    baseline['recall']=recall_score(ytest,
                    [1 for _ in range(len(ytest))])
    baseline['precision'] = precision_score(ytest,
                    [1 for _ in range(len(ytest))])
    baseline['roc'] = 0.5
    results = {}
    results['recall'] = recall_score(ytest, y_pred)
    results['precision'] = precision_score(ytest, y_pred)
    results['roc'] = roc_auc_score(ytest, probs)
    train_results = {}
    train_results['recall'] = recall_score(ytrain, train_predictions)
    train_results['precision'] = precision_score(ytrain, train_predictions)
    train_results['roc'] = roc_auc_score(ytrain, train_probs)
    for metric in ['recall', 'precision', 'roc']:  
          print(f"{metric.capitalize()} Baseline: {round(baseline[metric], 2)} Test: {round(results[metric], 2)} Train: {round(train_results[metric], 2)}")
     # Calculate false positive rates and true positive rates
    base_fpr, base_tpr, _ = roc_curve(ytest, [1 for _ in range(len(ytest))])
    model_fpr, model_tpr, _ = roc_curve(ytest, probs)
    plt.figure(figsize = (8, 6))
    plt.rcParams['font.size'] = 16
    # Plot both curves
    plt.plot(base_fpr, base_tpr, 'b', label = 'baseline')
    plt.plot(model_fpr, model_tpr, 'r', label = 'model')
    plt.legend();
    plt.xlabel('False Positive Rate');
    plt.ylabel('True Positive Rate'); plt.title('ROC Curves');
    plt.show();
evaluate_model(pred_lr,probs,train_predictions,train_probs)

from numpy import arange
from numpy import sqrt
from numpy import argmax
# apply threshold to positive probabilities to create labels
def to_labels(pos_probs, threshold):
	return (pos_probs >= threshold).astype('int')
# predict probabilities
yhat = model_LR.predict_proba(xtest)
# keep probabilities for the positive outcome only
probs = yhat[:, 1]
# define thresholds
thresholds = arange(0, 1, 0.01)
# evaluate each threshold
scores = [f1_score(ytest, to_labels(probs, t)) for t in thresholds]
# get best threshold
ix = argmax(scores)
print('Threshold=%.3f, F-Score=%.5f' % (thresholds[ix], scores[ix]))

y_pred = (model_LR.predict_proba(xtest)[:,1] >= 0.54).astype(int) 
#a = model_LR.predict_proba(xtest)
print(accuracy_score(y_pred,ytest))

print(confusion_matrix(ytest,pred_rf))
print(classification_report(ytest,pred_rf))


cm = confusion_matrix(ytest, y_pred)
plot_confusion_matrix(cm, classes = ['0 - Healthy','1 - Disease'],
                      title = 'Healthy_status Confusion Matrix')



#Future Importance
print(model_LR.feature_importances_)
print(f" There are {len(model_LR.feature_importances_)} features in total")


xtrain = pd.DataFrame(xtrain)
sns.barplot(x = xtrain.columns.tolist(), y = model_LR.feature_importances_)


feature_importances = list(zip(xtrain.columns, model_RF.feature_importances_))
feature_importances_ranked = sorted(feature_importances, key = lambda x: x[1], reverse = True)
[print('Feature: {:35} Importance: {}'.format(*pair)) for pair in feature_importances_ranked]


feature_names_25 = [i[0] for i in feature_importances_ranked[:18]]
y_ticks = np.arange(0, len(feature_names_25))
x_axis = [i[1] for i in feature_importances_ranked[:18]]
plt.figure(figsize = (10, 14))
plt.barh(feature_names_25, x_axis)   #horizontal barplot
plt.title('Random Forest Feature Importance (Top 18)',
          fontdict= {'fontname':'Comic Sans MS','fontsize' : 20})
plt.xlabel('Features',fontdict= {'fontsize' : 16})
plt.show()












