#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 18:30:13 2021

@author: downey
"""

# data = pd.read_csv("/Users/downey/Desktop/final_V.csv")
# data.isnull().sum()

# data.age_group = data.age_group.astype(str)
# data.Gender = data.Gender.astype(str)

# columns = data.columns.tolist()
# columns.remove("Healthy")
# columns.insert(0, "Healthy")



# y = data[["Healthy", "age_group", "Gender"]]
# print(y)


# x = data.drop(['Healthy', "age_group", "Gender"], axis = 1)
# min_max_scaler = preprocessing.MinMaxScaler()
# x = min_max_scaler.fit_transform(x)

# x = pd.DataFrame(x)

# data =  pd.concat([y, x], axis = 1)

# data.columns = columns


#print(voice.shape)

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

#sns.catplot("Healthy","Age", data = d1)


path = '/Users/downey/Desktop/final_II.csv'
data = np.loadtxt(path, dtype=float, delimiter=',', skiprows = 1)
d1 = pd.read_csv("/Users/downey/Desktop/final_V.csv")
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




xtrain, xtest, ytrain, ytest = train_test_split(x_smo, y_smo, test_size=0.30, random_state=10)

#-------------------Random Forest-----------------------------------
from sklearn.ensemble import RandomForestClassifier

time_start=time.time()

model_RF = RandomForestClassifier()
model_RF.fit(xtrain, ytrain)   
pred_rf = model_RF.predict(xtest)

time_end=time.time()
print('time cost',time_end-time_start,'s')


print(accuracy_score(pred_rf,ytest))
print(confusion_matrix(ytest,pred_rf))
print(classification_report(ytest,pred_rf))


#Evaluate the classifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve, f1_score


accuracy_score(ytest, pred_rf)
print(f"The accuracy of the model is {round(accuracy_score(ytest,pred_rf),3)*100} %")
# The accuracy of the model is 91.1%


train_probs = model_RF.predict_proba(xtrain)[:,1] 
probs = model_RF.predict_proba(xtest)[:, 1]
train_predictions = model_RF.predict(xtrain)

print(f'Train ROC AUC Score: {roc_auc_score(ytrain, train_probs)}')
print(f'Test ROC AUC  Score: {roc_auc_score(ytest, probs)}')


#ROC CURVE
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
evaluate_model(pred_rf,probs,train_predictions,train_probs)

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
plot_confusion_matrix(cm, classes = ['1 - Disease', '0 - Healthy'],
                      title = 'Healthy_status Confusion Matrix')


print(model_RF.feature_importances_)
print(f" There are {len(model_RF.feature_importances_)} features in total")


xtrain = pd.DataFrame(xtrain)
sns.barplot(x = xtrain.columns.tolist(), y = model_RF.feature_importances_)


feature_importances = list(zip(xtrain.columns, model_RF.feature_importances_))
feature_importances_ranked = sorted(feature_importances, key = lambda x: x[1], reverse = True)
[print('Feature: {:35} Importance: {}'.format(*pair)) for pair in feature_importances_ranked]


feature_names_25 = [i[0] for i in feature_importances_ranked[:25]]
y_ticks = np.arange(0, len(feature_names_25))
x_axis = [i[1] for i in feature_importances_ranked[:25]]
plt.figure(figsize = (10, 14))
plt.barh(feature_names_25, x_axis)   #horizontal barplot
plt.title('Random Forest Feature Importance (Top 25)',
          fontdict= {'fontname':'Comic Sans MS','fontsize' : 20})
plt.xlabel('Features',fontdict= {'fontsize' : 16})
plt.show()


from pprint import pprint
print('Parameters currently in use:\n')
pprint(model_RF.get_params())


from sklearn.model_selection import RandomizedSearchCV
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 700, num = 50)]
max_features = ['auto', 'log2']  # Number of features to consider at every split
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]   # Maximum number of levels in tree
max_depth.append(None)
min_samples_split = [2, 5, 10]  # Minimum number of samples required to split a node
min_samples_leaf = [1, 4, 10]    # Minimum number of samples required at each leaf node
bootstrap = [True, False]       # Method of selecting samples for training each tree
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'max_leaf_nodes': [None] + list(np.linspace(10, 50, 500).astype(int)),
               'bootstrap': bootstrap}



# Create base model to tune
rf = RandomForestClassifier(oob_score=True)
# Create random search model and fit the data
rf_random = RandomizedSearchCV(
                        estimator = rf,
                        param_distributions = random_grid,
                        n_iter = 100, cv = 3,
                        verbose=2, random_state=100, 
                        scoring='roc_auc')
rf_random.fit(xtrain, ytrain)
print(rf_random.best_params_)



from sklearn.pipeline import make_pipeline
# Use the best model after tuning
best_model = rf_random.best_estimator_

best_model.fit(xtrain, ytrain)
y_pred_best_model = best_model.predict(xtest)


train_rf_predictions = best_model.predict(xtrain)
train_rf_probs = best_model.predict_proba(xtrain)[:, 1]
rf_probs = best_model.predict_proba(xtest)[:, 1]
# Plot ROC curve and check scores
evaluate_model(y_pred_best_model, rf_probs, train_rf_predictions, train_rf_probs)


plot_confusion_matrix(confusion_matrix(ytest, y_pred_best_model), classes = ['0 - Stay', '1 - Exit'],
title = 'Exit_status Confusion Matrix')















