#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 11:40:18 2021

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
from pprint import pprint

#sns.catplot("Healthy","Age", data = d1)


path = '/Users/downey/Desktop/final_II.csv'
data = np.loadtxt(path, dtype=float, delimiter=',', skiprows = 1)
d1 = pd.read_csv("/Users/downey/Desktop/final_V.csv")
data.shape[1]

#Get Columns name
columns = d1.columns.tolist()
columns.remove("Healthy")
# columns.remove("age_group")
# columns.remove("Gender")

# columns.append("Age_2")
# columns.append("Age_3")
# columns.append("Male")

x,y = np.split(data, (data.shape[1] - 1,), axis = 1)


#----------------------------- PairPlot --------------------------------------
# data = pd.DataFrame(data)
# data.columns = columns
# p_d = data.iloc[: , [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]]
# #sns.pairplot(p_d)

# g = sns.pairplot(p_d, hue="Healthy", markers=["o", "s"], palette='Dark2')
# g.map_lower(sns.kdeplot, levels=4, color=".2")
h = d1[["Healthy"]]
d1 = d1.iloc[:, [0,1,2,3,4,5,6]]

my = pd.concat([h, d1], axis = 1)

g = sns.pairplot(my, hue="Healthy", markers=["o", "s"], palette='Dark2')
g.map_lower(sns.kdeplot, levels=4, color=".2")



#-------------------------- Dummy Vairable --------------------------------
#Dummy Variables
# df = pd.DataFrame(data = x)
# df = pd.get_dummies(df, columns = [0,1], drop_first = True)
#--------------------------------------------------------------------------

#MINMAX SCaler
#x = np.array(df)
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


#-----------------------------------------------------------------------------
# x_smo.loc[x_smo.Age_2 < 0.5, "Age_2"] = 0
# x_smo.loc[x_smo.Age_2 > 0.5, "Age_2"] = 1

# x_smo.loc[x_smo.Age_3 < 0.5, "Age_3"] = 0
# x_smo.loc[x_smo.Age_3 > 0.5, "Age_3"] = 1

# x_smo.loc[x_smo.Male < 0.5, "Male"] = 0
# x_smo.loc[x_smo.Male > 0.5, "Male"] = 1
#-----------------------------------------------------------------------------


xtrain, xtest, ytrain, ytest = train_test_split(x_smo, y_smo, test_size=0.30, random_state=10)

print(Counter(ytest['Healthy']))
# 0 - 15 (Healthy)
# 1 - 11 (Disease)

# 5-fold cross-validation
score_svm = 0
score_rf = 0
score_lr = 0
score_nb = 0
score_gbdt = 0
kf = KFold(n_splits=10, shuffle=True, random_state = 1)
for train, test in kf.split(data):
    train = data[train]
    test = data[test]
    xtrain, ytrain = np.split(train, (data.shape[1] - 1,), axis=1)
    xtest, ytest = np.split(test, (data.shape[1] - 1,), axis=1)


    #clf = svm.SVC(C=128, kernel='rbf', gamma=0.0001, decision_function_shape='ovr')
    svc = svm.SVC()
    svc.fit(xtrain, ytrain.ravel())
    ypredict = svc.predict(xtest)
    score_svm += svc.score(xtest, ytest)

    #rfc = RandomForestClassifier(n_estimators=250, max_depth=None, min_samples_split=2, bootstrap=True)
    rfc = RandomForestClassifier()
    rfc = rfc.fit(xtrain, ytrain.ravel())
    #print(rfc.score(xtest,ytest))
    score_rf += rfc.score(xtest, ytest)

    #clf = LogisticRegression(penalty='l2', C=1.0)
    clf = LogisticRegression()
    clf = clf.fit(xtrain, ytrain.ravel())
    score_lr += clf.score(xtest, ytest)

    gnb = GaussianNB()
    gnb = gnb.fit(xtrain, ytrain.ravel())
    score_nb += gnb.score(xtest, ytest)

    model = GradientBoostingClassifier()
    model = model.fit(xtrain, ytrain.ravel())
    score_gbdt += model.score(xtest, ytest)


#pprint(rfc.get_params())

#-----------------------------------------------------------------------------
def plot_confusion_matrix(cm, classes=None, title='Confusion matrix'):
    """Plots a confusion matrix."""
    if classes is not None:
        sns.heatmap(cm, xticklabels=classes, yticklabels=classes, vmin=0., vmax=1., annot=True, annot_kws={'size':50})
    else:
        sns.heatmap(cm, vmin=0., vmax=1.)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

rf = RandomForestClassifier(n_estimators=100, criterion='entropy')
rf.fit(xtrain, ytrain)
prediction_test = rf.predict(X=xtest)


# Accuracy on Test
print("Training Accuracy is: ", rf.score(xtrain, ytrain))
# Accuracy on Train
print("Testing Accuracy is: ", rf.score(xtest, ytest))

# Confusion Matrix
cm = confusion_matrix(ytest, prediction_test)
cm_norm = cm/cm.sum(axis=1)[:, np.newaxis]
plt.figure()
plot_confusion_matrix(cm_norm, classes=rf.classes_)

y_pred = rf.predict(xtest)
confusion_matrix(y_pred,ytest)

ytrain = np.array(ytrain)
from itertools import product
n_estimators = [10, 25, 50, 100, 150, 200, 250]
max_features = ["auto", "sqrt", "log2"]
max_depths = [None, 2, 3, 4, 5]
min_samples_leaf = [1, 2, 3, 4]
for f, d, i, k in product(max_features, max_depths, n_estimators, min_samples_leaf): # with product we can iterate through all possible combinations
    rf = RandomForestClassifier(n_estimators=i, 
                                criterion='entropy', 
                                max_features=f, 
                                max_depth=d, 
                                min_samples_leaf = k,
                                n_jobs=2,
                                random_state=1337)
    rf.fit(xtrain, ytrain.ravel())
    prediction_test = rf.predict(X=xtest)
    print('Classification accuracy on test set with max features = {} and max_depth = {}: {:.3f}'.format(f, d, accuracy_score(ytest,prediction_test)))
    cm = confusion_matrix(ytest, prediction_test)
    cm_norm = cm/cm.sum(axis=1)[:, np.newaxis]
    plt.figure()
    plot_confusion_matrix(cm_norm, classes=rf.classes_,
    title='Confusion matrix accuracy on test set with max features = {} and max_depth = {}: {:.3f}'.format(f, d, accuracy_score(ytest,prediction_test)))
    
    X_train_prediction = rf.predict(xtrain)
    training_data_accuracy = accuracy_score(ytrain, X_train_prediction)
    print("Accuracy Score of training data: ", training_data_accuracy)
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#Method Two

#-----------------------------------------------------------------------------
    

time_start=time.time()

model_RF = RandomForestClassifier()
model_RF.fit(xtrain, ytrain)   
pred_rf = model_RF.predict(xtest)

time_end=time.time()
print('time cost',time_end-time_start,'s')


print(accuracy_score(pred_rf,ytest))
print(confusion_matrix(ytest,pred_rf))
print(classification_report(ytest,pred_rf))


#Accuracy on Training / Testing
X_train_prediction = model_RF.predict(xtrain)
training_data_accuracy = accuracy_score(ytrain, X_train_prediction)
print("Accuracy Score of training data: ", training_data_accuracy)

#accuracy score on training data
X_test_prediction = model_RF.predict(xtest)
testing_data_accuracy = accuracy_score(ytest, X_test_prediction)
print("Accuracy Score of test data: ", testing_data_accuracy)

#Plot Confusion Matrix
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


# from sklearn.model_selection import RandomizedSearchCV
# #Tune Parameters
# # Number of trees in random forest
# n_estimators = [int(x) for x in np.linspace(start = 100, stop = 700, num = 50)]
# # Number of features to consider at every split
# max_features = ['auto', 'sqrt']
# # Maximum number of levels in tree
# max_depth = [1, 5, 10, 20, 50, 75, 100, 150, 200]
# max_depth.append(None)
# # Minimum number of samples required to split a node
# # min_samples_split = [int(x) for x in np.linspace(start = 2, stop = 10, num = 9)]
# min_samples_split = [1, 2, 5, 10, 15, 20, 30]
# # Minimum number of samples required at each leaf node
# min_samples_leaf = [1, 2, 3, 4]
# # Method of selecting samples for training each tree
# bootstrap = [True, False]
# # Criterion
# criterion=['gini', 'entropy']
# random_grid = {'n_estimators': n_estimators,
#                'max_features': max_features,
#                'max_depth': max_depth,
#                'min_samples_split': min_samples_split,
#                'min_samples_leaf': min_samples_leaf,
#                'bootstrap': bootstrap,
#                'criterion': criterion}


# rf_base = RandomForestClassifier()
# rf_random = RandomizedSearchCV(estimator = rf_base,
#                                param_distributions = random_grid,
#                                n_iter = 30, cv = 5,
#                                verbose=2,
#                                random_state=42, n_jobs = 4)
# rf_random.fit(xtrain, ytrain)
# print(rf_random.best_params_)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
model = RandomForestClassifier()
n_estimators = [10, 25, 50, 100, 150, 200, 250]
max_features = ['sqrt', 'log2']
max_depth=[None, 5, 10]
criterion=["gini", "entropy"]
# define grid search
grid = dict(n_estimators=n_estimators,max_features=max_features,max_depth=max_depth,criterion=criterion)
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(x_smo, y_smo)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


#model_RF = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2,bootstrap=True)
model_RF_1 = RandomForestClassifier(n_estimators=150, max_depth = None, max_features = "sqrt", bootstrap=True)
model_RF_1.fit(xtrain, ytrain)
pred_RF_1 = model_RF_1.predict(xtest)
print(accuracy_score(pred_RF_1,ytest))

print(confusion_matrix(ytest,pred_RF_1))
print(classification_report(ytest,pred_RF_1))

cm = confusion_matrix(ytest, pred_RF_1)
plot_confusion_matrix(cm, classes = ['0 - Healthy','1 - Disease'],
                      title = 'Healthy_status Confusion Matrix')




from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve, f1_score
train_probs = model_RF_1.predict_proba(xtrain)[:,1] 
probs = model_RF_1.predict_proba(xtest)[:, 1]
train_predictions = model_RF_1.predict(xtrain)

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
evaluate_model(pred_RF_1,probs,train_predictions,train_probs)


#----------------------------
from numpy import sqrt
from numpy import argmax
# predict probabilities
yhat = model_RF_1.predict_proba(xtest)
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
yhat = model_RF_1.predict_proba(xtest)
# keep probabilities for the positive outcome only
probs = yhat[:, 1]
# define thresholds
thresholds = arange(0, 1, 0.01)
# evaluate each threshold
scores = [f1_score(ytest, to_labels(probs, t)) for t in thresholds]
# get best threshold
ix = argmax(scores)
print('Threshold=%.3f, F-Score=%.5f' % (thresholds[ix], scores[ix]))


y_pred = (model_RF_1.predict_proba(xtest)[:,1] >= 0.49).astype(int) 
#a = model_LR.predict_proba(xtest)
print(accuracy_score(y_pred,ytest))
print(confusion_matrix(ytest,pred_rf))
print(classification_report(ytest,pred_rf))

cm = confusion_matrix(ytest, y_pred)
plot_confusion_matrix(cm, classes = ['0 - Healthy','1 - Disease'],
                      title = 'Healthy_status Confusion Matrix')

print(y_pred)
print(ytest)


# prob_preds = clf.predict_proba(X)
# threshold = 0.11 # define threshold here
# preds = [1 if prob_preds[i][1]> threshold else 0 for i in range(len(prob_preds))]


ytrain_pred = model_RF_1.predict_proba(xtrain)
print('RF train roc-auc: {}'.format(roc_auc_score(ytrain, ytrain_pred[:,1])))
ytest_pred = model_RF_1.predict_proba(xtest)
print('RF test roc-auc: {}'.format(roc_auc_score(ytest, ytest_pred[:,1])))


from sklearn.metrics import accuracy_score
accuracy_ls = []
for thres in thresholds:
    y_pred = np.where(ytest_pred[:,1]>thres,1,0)
    accuracy_ls.append(accuracy_score(ytest, y_pred, normalize=True))
    
accuracy_ls = pd.concat([pd.Series(thresholds), pd.Series(accuracy_ls)],
                        axis=1)
accuracy_ls.columns = ['thresholds', 'accuracy']
accuracy_ls.sort_values(by='accuracy', ascending=False, inplace=True)
print(accuracy_ls)

def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()
plot_roc_curve(fpr,tpr)







