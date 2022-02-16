#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 16:41:29 2021

@author: downey
"""

#------------------------------- Packages -------------------------------------
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
from sklearn.preprocessing import StandardScaler

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

#----------------------------- PairPlot --------------------------------------
# data = pd.DataFrame(data)
# data.columns = columns
# p_d = data.iloc[: , [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]]
# #sns.pairplot(p_d)

# g = sns.pairplot(p_d, hue="Healthy", markers=["o", "s"], palette='Dark2')
# g.map_lower(sns.kdeplot, levels=4, color=".2")

#####################################
# g = sns.pairplot(d1, hue="Healthy", markers=["o", "s"], palette='Dark2')
# g.map_lower(sns.kdeplot, levels=4, color=".2")
#####################################

#Heatmap
#df_impo = pd.concat([voice_df['Healthy'], voice], axis = 1)
#print(df_impo.shape)

#####################################
# sns.heatmap(d1.corr())

# plt.figure(figsize=(16, 6))
# mask = np.triu(np.ones_like(d1.corr(), dtype=np.bool))
# heatmap = sns.heatmap(d1.corr(), mask = mask, vmin=-1, vmax=1, annot=True, cmap = "BrBG")
# heatmap.set_title('The correlation heatmap between attributes', fontdict={'fontsize':18}, pad=16)
#######################################


#------------------------------ MinMaxScaler ---------------------------------
######################
#MINMAX SCaler
min_max_scaler = preprocessing.MinMaxScaler()
x = min_max_scaler.fit_transform(x)

data = np.c_[x, y]
######################

# # the scaler object (model)
# scaler = StandardScaler()
# # fit and transform the data
# x = scaler.fit_transform(x) 
# data = np.c_[x, y]

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
xtrain, xtest, ytrain, ytest = train_test_split(x_smo, y_smo, test_size=0.30, random_state=14)
print(Counter(ytest['Healthy']))
ytrain = np.array(ytrain)
# Disease(0) - 14
# Healthy(1) - 12
xtrain.shape
xtest.shape

#------------------------------------ Default RF ------------------------------
#Check the default parameters setting
rf = RandomForestClassifier(random_state = 1)
print('Parameters currently in use:\n')
pprint(rf.get_params())

#Default RF
# random forest model creation
rfc = RandomForestClassifier()
rfc.fit(xtrain,ytrain)
rfc_predict = rfc.predict(xtest)

print(accuracy_score(rfc_predict,ytest)) #0.7307692307692307
print(confusion_matrix(ytest,rfc_predict))
print(classification_report(ytest,rfc_predict))

rfc_cv_score = cross_val_score(rfc, x_smo, y_smo, cv=10, scoring="roc_auc")
print(np.average(rfc_cv_score)) #0.78125 0.7925

print("=== Confusion Matrix ===")
print(confusion_matrix(ytest,rfc_predict))
print('\n')
print("=== Classification Report ===")
print(classification_report(ytest, rfc_predict))
print('\n')
print("=== Overall Accuracy Score ===")
print(accuracy_score(rfc_predict,ytest))
print('\n')
print("=== Mean AUC Score ===")
print("Mean AUC Score - Random Forest: ", rfc_cv_score.mean())

X_train_prediction = rfc.predict(xtrain)
training_data_accuracy = accuracy_score(ytrain, X_train_prediction)
print("Accuracy Score of training data: ", training_data_accuracy)

#accuracy score on training data
X_test_prediction = rfc.predict(xtest)
testing_data_accuracy = accuracy_score(ytest, X_test_prediction)
print("Accuracy Score of test data: ", testing_data_accuracy)

#------------------------------------ RF Tunning ------------------------------
#------------------------------------ Nested CV -------------------------------
from sklearn.model_selection import RandomizedSearchCV
from numpy import mean
from numpy import std
# configure the cross-validation procedure
cv_outer = KFold(n_splits=5, shuffle=True, random_state=1)
#parameters Options
space = dict()
space['n_estimators'] = [100, 150, 200, 250, 500]
space['max_features'] = ["auto", "sqrt"]
space["max_depth"] = [5, 10, 15, 20, 25]
space["max_depth"].append(None)
space["min_samples_split"] = [3, 5, 10, 15]
space["min_samples_leaf"] = [2, 5, 10]
space['max_leaf_nodes'] = [2, 5, 10]

# enumerate splits
outer_results = list()
for train_ix, test_ix in cv_outer.split(x_smo):
    # split data
    X_train, X_test = x_smo.iloc[train_ix, :], x_smo.iloc[test_ix, :]
    y_train, y_test = y_smo.iloc[train_ix], y_smo.iloc[test_ix]
	# configure the cross-validation procedure
    cv_inner = KFold(n_splits=3, shuffle=True, random_state=25)
	# define the model
    model = RandomForestClassifier(random_state=1)
	# define search
    search = RandomizedSearchCV(model, space, scoring='roc_auc', cv=cv_inner, refit=True, n_jobs=-1, n_iter = 100, random_state=25)
	# execute search
    result = search.fit(X_train, y_train)
	# get the best performing model fit on the whole training set
    best_model = result.best_estimator_
	# evaluate model on the hold out dataset
    yhat = best_model.predict(X_test)
	# evaluate the model
    acc = accuracy_score(y_test, yhat)
	# store the result
    outer_results.append(acc)
	# report progress
    print('>acc=%.3f, est=%.3f, cfg=%s' % (acc, result.best_score_, result.best_params_))
# summarize the estimated performance of the model
print(outer_results)
print('Accuracy: %.3f (%.3f)' % (mean(outer_results), std(outer_results)))


#------------------------------ Only RandomiszedCV ----------------------------

rcv_model = RandomForestClassifier(random_state=1)

random = RandomizedSearchCV(estimator = rcv_model, param_distributions = space, n_iter = 100, cv = 5, verbose=2, random_state=25, n_jobs = -1)
# Fit the random search model
start = random.fit(xtrain, ytrain)
print(start.best_params_)
my_model = start.best_estimator_
# print('Parameters currently in use:\n')
#pprint(my_model.get_params())
y_p = my_model.predict(xtest)
print(accuracy_score(y_p,ytest)) #0.7307692307692307
print(confusion_matrix(ytest,y_p))
print(classification_report(ytest,y_p))

rfc_cv_score = cross_val_score(my_model, x_smo, y_smo, cv=10, scoring="roc_auc")
print(np.average(rfc_cv_score))



#------------------------------------ Nested CV -------------------------------
rfc = RandomForestClassifier(n_estimators= 250, 
                              min_samples_split = 3, 
                              min_samples_leaf=5, 
                              max_leaf_nodes=5,
                              max_features = "auto",
                              max_depth=None,
                              random_state = 1)

rfc = RandomForestClassifier(n_estimators= 500, 
                              min_samples_split = 3, 
                              min_samples_leaf=3, 
                              max_leaf_nodes=10,
                              max_features = "auto",
                              max_depth=None,
                              random_state = 1)

rfc.fit(xtrain,ytrain)
rfc_predict = rfc.predict(xtest)
print(accuracy_score(rfc_predict,ytest))
print(confusion_matrix(ytest,rfc_predict))
print(classification_report(ytest,rfc_predict))

rfc_cv_score = cross_val_score(rfc, x_smo, y_smo, cv=10, scoring="roc_auc")
print(np.average(rfc_cv_score))

a = rfc.predict_proba(xtest)
a = pd.DataFrame(a)
b = ytest
b = b.reset_index(drop=True)
c = rfc_predict
c = pd.DataFrame(c)

p_table = pd.concat([a, b, c], axis = 1, join='inner')



#---------------------------- Confusion Matrix --------------------------------
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
cm = confusion_matrix(ytest, rfc_predict)
plot_confusion_matrix(cm, classes = ['0 - Disease','1 - Healthy'],
                      title = 'Healthy_status Confusion Matrix')


#------------------------------- ROC Cruve -------------------------------------
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve, f1_score
train_probs = rfc.predict_proba(xtrain)[:,1] 
probs = rfc.predict_proba(xtest)[:, 1]
train_predictions = rfc.predict(xtrain)

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
evaluate_model(rfc_predict,probs,train_predictions,train_probs)



from numpy import sqrt
from numpy import argmax
# predict probabilities
yhat = rfc.predict_proba(xtest)
# keep probabilities for the positive outcome only
yhat = yhat[:, 1]
# calculate roc curves
fpr, tpr, thresholds = roc_curve(ytest, yhat)
# calculate the g-mean for each threshold
gmeans = sqrt(tpr * (1-fpr))
# locate the index of the largest g-mean
ix = argmax(gmeans)
print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))
# plot the roc curve for the model
pyplot.plot([0,1], [0,1], linestyle='--', label='No Skill')
pyplot.plot(fpr, tpr, marker='.', label='Logistic')
pyplot.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best')
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
pyplot.legend()
# show the plot
pyplot.show()

from numpy import arange
# apply threshold to positive probabilities to create labels
def to_labels(pos_probs, threshold):
	return (pos_probs >= threshold).astype('int')
# predict probabilities
yhat = rfc.predict_proba(xtest)
# keep probabilities for the positive outcome only
probs = yhat[:, 1]
# define thresholds
thresholds = arange(0, 1, 0.01)
# evaluate each threshold
scores = [f1_score(ytest, to_labels(probs, t)) for t in thresholds]
# get best threshold
ix = argmax(scores)
print('Threshold=%.3f, F-Score=%.5f' % (thresholds[ix], scores[ix]))

y_pred = (rfc.predict_proba(xtest)[:,1] >= 0.52).astype(int) 
#a = model_LR.predict_proba(xtest)
print(accuracy_score(y_pred,ytest))


#------------------------------------------------------------------------------
#--------------------------- Feature Importance -------------------------------
#Future Importance
print(rfc.feature_importances_)
print(f" There are {len(rfc.feature_importances_)} features in total")


xtrain = pd.DataFrame(xtrain)
sns.barplot(x = xtrain.columns.tolist(), y = rfc.feature_importances_)


feature_importances = list(zip(xtrain.columns, rfc.feature_importances_))
feature_importances_ranked = sorted(feature_importances, key = lambda x: x[1], reverse = True)
[print('Feature: {:14} Importance: {}'.format(*pair)) for pair in feature_importances_ranked]


feature_names_25 = [i[0] for i in feature_importances_ranked[:18]]
y_ticks = np.arange(0, len(feature_names_25))
x_axis = [i[1] for i in feature_importances_ranked[:18]]
plt.figure(figsize = (10, 14))
plt.barh(feature_names_25, x_axis)   #horizontal barplot
plt.title('Random Forest Feature Importance (Top 18)',
          fontdict= {'fontname':'Comic Sans MS','fontsize' : 20})
plt.xlabel('Features',fontdict= {'fontsize' : 16})
plt.show()

from sklearn.inspection import permutation_importance
result = permutation_importance(
    rfc, xtest, ytest, n_repeats=10, random_state=42, n_jobs=2
)
sorted_idx = result.importances_mean.argsort()

fig, ax = plt.subplots()
ax.boxplot(
    result.importances[sorted_idx].T, vert=False, labels=xtest.columns[sorted_idx]
)
ax.set_title("Permutation Importances (test set)")
fig.tight_layout()
plt.show()


result = permutation_importance(
    rfc, xtrain, ytrain, n_repeats=10, random_state=42, n_jobs=2
)
sorted_idx = result.importances_mean.argsort()

fig, ax = plt.subplots()
ax.boxplot(
    result.importances[sorted_idx].T, vert=False, labels=xtrain.columns[sorted_idx]
)
ax.set_title("Permutation Importances (train set)")
fig.tight_layout()
plt.show()

#--------------------------- New Dataset -------------------------------------


cs = x_smo.columns.tolist()
x_smo = x_smo.drop(["Mean.Period.msec."], axis = 1)
xtrain, xtest, ytrain, ytest = train_test_split(x_smo, y_smo, test_size=0.30)

rfc = RandomForestClassifier(n_estimators= 500, 
                              min_samples_split = 3, 
                              min_samples_leaf=3, 
                              max_leaf_nodes=10,
                              max_features = "auto",
                              max_depth=None,
                              random_state = 1)

rfc.fit(xtrain,ytrain)
rfc_predict = rfc.predict(xtest)
print(accuracy_score(rfc_predict,ytest))
print(confusion_matrix(ytest,rfc_predict))
print(classification_report(ytest,rfc_predict))


# cs = x_smo.columns.tolist()
# x_smo = x_smo.drop(["vFo"], axis = 1)
# xtrain, xtest, ytrain, ytest = train_test_split(x_smo, y_smo, test_size=0.30, random_state=14)

#------------------------------- ROC Cruve -------------------------------------
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve, f1_score
train_probs = rfc.predict_proba(xtrain)[:,1] 
probs = rfc.predict_proba(xtest)[:, 1]
train_predictions = rfc.predict(xtrain)

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
evaluate_model(rfc_predict,probs,train_predictions,train_probs)

from numpy import arange
# apply threshold to positive probabilities to create labels
def to_labels(pos_probs, threshold):
	return (pos_probs >= threshold).astype('int')
# predict probabilities
yhat = rfc.predict_proba(xtest)
# keep probabilities for the positive outcome only
probs = yhat[:, 1]
# define thresholds
thresholds = arange(0, 1, 0.01)
# evaluate each threshold
scores = [f1_score(ytest, to_labels(probs, t)) for t in thresholds]
# get best threshold
ix = argmax(scores)
print('Threshold=%.3f, F-Score=%.5f' % (thresholds[ix], scores[ix]))

y_pred = (rfc.predict_proba(xtest)[:,1] >= 0.45).astype(int) 
#a = model_LR.predict_proba(xtest)
print(accuracy_score(y_pred,ytest))
print(confusion_matrix(ytest,rfc_predict))
print(classification_report(ytest,rfc_predict))

#-------------------------------------------------------------------------------










# Create base model to tune
model = RandomForestClassifier(random_state=1)
# Create random search model and fit the data
rf_random = RandomizedSearchCV(
                        estimator = model,
                        param_distributions = space,
                        n_iter = 100, cv = 3,
                        verbose=2, random_state=100, 
                        scoring='roc_auc')
rf_random.fit(xtrain, ytrain)
print(rf_random.best_params_)

rfc = RandomForestClassifier(n_estimators= 500, 
                              min_samples_split = 3, 
                              min_samples_leaf=2, 
                              max_leaf_nodes=10,
                              max_features = "auto",
                              max_depth = 5,
                              random_state = 1)
rfc.fit(xtrain,ytrain)
rfc_predict = rfc.predict(xtest)
rfc_cv_score = cross_val_score(rfc, x_smo, y_smo, cv=10, scoring="roc_auc")
print(np.average(rfc_cv_score))





from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
y_smo = np.array(y_smo)
model = RandomForestClassifier()
# Number of trees in random forest
n_estimators = [10, 25, 50, 100, 150, 200, 250]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [5, 10, 15]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 3, 4, 5, 6]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 3, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# define grid search
grid = dict(n_estimators=n_estimators,max_features=max_features,max_depth=max_depth,min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaf,bootstrap=bootstrap)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(x_smo, y_smo.ravel())
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
    
    
    


#-------------------------------------------------------------------------------
from sklearn import model_selection



rfc = RandomForestClassifier(n_estimators=500, min_samples_split = 2, max_depth=None, max_features='sqrt',min_samples_leaf=4,bootstrap=False,random_state = 36)
rfc.fit(xtrain,ytrain)
rfc_predict = rfc.predict(xtest)
rfc_cv_score = cross_val_score(rfc, x_smo, y_smo, cv=10, scoring='roc_auc')
print("=== Confusion Matrix ===")
print(confusion_matrix(ytest, rfc_predict))
print('\n')
print("=== Classification Report ===")
print(classification_report(ytest, rfc_predict))
print('\n')
print("=== Overall Accuracy ===")
print(accuracy_score(rfc_predict,ytest))
print('\n')
print("=== Mean AUC Score ===")
print("Mean AUC Score - Random Forest: ", rfc_cv_score.mean())


X_train_prediction = rfc.predict(xtrain)
training_data_accuracy = accuracy_score(ytrain, X_train_prediction)
print("Accuracy Score of training data: ", training_data_accuracy)

#accuracy score on training data
X_test_prediction = rfc.predict(xtest)
testing_data_accuracy = accuracy_score(ytest, X_test_prediction)
print("Accuracy Score of test data: ", testing_data_accuracy)


#------------------------------------------------------------------------------
from sklearn.metrics import roc_curve, precision_recall_curve, auc, make_scorer, recall_score, accuracy_score, precision_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
clf = RandomForestClassifier(n_jobs=-1)

param_grid = {
    'min_samples_split': [3, 5, 10], 
    'n_estimators' : [100, 200, 300],
    'max_depth': [3, 5, 15],
    'max_features': [3, 5, 10, 20]
}

scorers = {
    'precision_score': make_scorer(precision_score),
    'recall_score': make_scorer(recall_score),
    'accuracy_score': make_scorer(accuracy_score)
}

def grid_search_wrapper(refit_score='precision_score'):
    """
    fits a GridSearchCV classifier using refit_score for optimization
    prints classifier performance metrics
    """
    skf = StratifiedKFold(n_splits=10)
    grid_search = GridSearchCV(clf, param_grid, scoring=scorers, refit=refit_score,
                           cv=skf, return_train_score=True, n_jobs=-1)
    grid_search.fit(xtrain, ytrain)

    # make the predictions
    y_pred = grid_search.predict(xtest)

    print('Best params for {}'.format(refit_score))
    print(grid_search.best_params_)

    # confusion matrix on the test data.
    print('\nConfusion matrix of Random Forest optimized for {} on the test data:'.format(refit_score))
    print(pd.DataFrame(confusion_matrix(ytest, y_pred),
                 columns=['pred_neg', 'pred_pos'], index=['neg', 'pos']))
    return grid_search

grid_search_clf = grid_search_wrapper(refit_score='precision_score')

results = pd.DataFrame(grid_search_clf.cv_results_)
results = results.sort_values(by='mean_test_precision_score', ascending=False)
a = results[['mean_test_precision_score', 'mean_test_recall_score', 'mean_test_accuracy_score', 'param_max_depth', 'param_max_features', 'param_min_samples_split', 'param_n_estimators']].round(3).head()
#------------------------------------------------------------------------------
from sklearn.model_selection import RepeatedStratifiedKFold
model = RandomForestClassifier()
n_estimators = [10, 100, 300, 500]
max_features = ['sqrt', 'log2']
# define grid search
grid = dict(n_estimators=n_estimators,max_features=max_features)
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

























