#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 17:05:55 2021

@author: downey
"""
#-------------------------------- Packages Import -----------------------------
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
from sklearn.model_selection import train_test_split
import math
from sklearn.model_selection import KFold,LeaveOneOut,LeavePOut,ShuffleSplit
from sklearn import preprocessing
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import seaborn as sns
from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.metrics import plot_confusion_matrix

from imblearn.over_sampling import SMOTE
from matplotlib import pyplot

from pprint import pprint


#--------------------------- Data Import --------------------------------------
path = '/Users/downey/Desktop/aequitas.csv'
data = np.loadtxt(path, dtype=float, delimiter=',', skiprows = 1)
d1 = pd.read_csv("/Users/downey/Desktop/aequitas.csv")
data.shape[1]

print(Counter(d1['Healthy']))
#Counter({1: 42, 0: 27})

#Get Columns name
columns = d1.columns.tolist()
columns.remove("Healthy")
columns.remove("Age_group")
columns.remove("Gender")

columns.append("Age_2")
columns.append("Age_3")
columns.append("Male")


#-------------------------- Dummy Variables -----------------------------------
x,y = np.split(data, (data.shape[1] - 1,), axis = 1)
df = pd.DataFrame(data = x)
df = pd.get_dummies(df, columns = [0,1], drop_first = True)

#------------------------------ MINMAX SCaler ---------------------------------
x = np.array(df)
min_max_scaler = preprocessing.MinMaxScaler()
x = min_max_scaler.fit_transform(x)

data = np.c_[x, y]
#data = pd.DataFrame(data)


#----------------------------- SMOTE -----------------------------------------
smo = SMOTE(random_state = 25)
x_smo, y_smo = smo.fit_resample(x, y)
data = np.c_[x_smo, y_smo]

#x_smo Name
x_smo = pd.DataFrame(x_smo)
x_smo.columns = columns
#y_smo Name
y_smo = pd.DataFrame(y_smo)
y_smo.columns = ["Healthy"]




#---------------------------- Some Adjustment for SMOTE ----------------------
x_smo.loc[x_smo.Age_2 < 0.5, "Age_2"] = 0
x_smo.loc[x_smo.Age_2 > 0.5, "Age_2"] = 1

x_smo.loc[x_smo.Age_3 < 0.5, "Age_3"] = 0
x_smo.loc[x_smo.Age_3 > 0.5, "Age_3"] = 1

x_smo.loc[x_smo.Male < 0.5, "Male"] = 0
x_smo.loc[x_smo.Male > 0.5, "Male"] = 1

data = pd.concat([x_smo,y_smo], axis = 1, join='inner')
data = np.array(data)


#---------------------------- Model Selection ---------------------------------
score_svm = 0
score_rf = 0
score_lr = 0
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


#---------------------------- Train/Test SPlit -------------------------------
xtrain, xtest, ytrain, ytest = train_test_split(x_smo, y_smo, test_size=0.30, random_state=14)



#---------------------------- Default LR -------------------------------------
lr = LogisticRegression()
lr.fit(xtrain,ytrain)
lr_predict = lr.predict(xtest)

print(accuracy_score(lr_predict,ytest)) 
print(confusion_matrix(ytest,lr_predict))
print(classification_report(ytest,lr_predict))


#----------------------- Hyperparameters Tunninig -----------------------------
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
model = LogisticRegression()
solvers = ['newton-cg', 'lbfgs', 'liblinear']
penalty = ['l2']
c_values = [100, 10, 1.0, 0.1, 0.01]
# define grid search
grid = dict(solver=solvers,penalty=penalty,C=c_values)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=10)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)

grid_result = grid_search.fit(x_smo, y_smo)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

my_model = grid_result.best_estimator_
y_p = my_model.predict(xtest)
print(accuracy_score(y_p,ytest))
print(confusion_matrix(ytest,y_p))
print(classification_report(ytest,y_p))


#Prepare for Aequitas (Score Column)
score = my_model.predict(x)
print(accuracy_score(score,y))

score = pd.DataFrame(score)
y = pd.DataFrame(y)
ccc = pd.concat([score,y], axis = 1, join='inner')

ccc.to_csv (r'/Users/downey/Desktop/score.csv', index = False, header=True)


#------------------------------- ROC Cruve -------------------------------------
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve, f1_score
train_probs = my_model.predict_proba(xtrain)[:,1] 
probs = my_model.predict_proba(xtest)[:, 1]
train_predictions = my_model.predict(xtrain)

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
evaluate_model(y_p,probs,train_predictions,train_probs)

#------------------------------- Aequitas -------------------------------------
import aequitas
from aequitas.group import Group
from aequitas.plotting import Plot

a_table = pd.read_csv("/Users/downey/Desktop/score.csv")
my_ae = a_table[["score", "label_value", "Age_group", "Gender"]]

#Check Data types
dataTypeSeries = my_ae.dtypes
print('Data type of each column of Dataframe :')
print(dataTypeSeries)


aq_palette = sns.diverging_palette(225, 35, n=2)
by_age_group = sns.countplot(x="Age_group", hue="score", data=my_ae[my_ae.Age_group.isin(['Age more than 74', 'Age between 59 to 74', 'Age between 39 to 59'])], palette=aq_palette)

by_gender = sns.countplot(x="Gender", hue="score", data=my_ae, palette=aq_palette)

#----------------------------------- Aequitas ---------------------------------

my_ae['Age_group'] = my_ae['Age_group'].astype(str)
my_ae['Gender'] = my_ae['Gender'].astype(str)

g = Group()
xtab, _ = g.get_crosstabs(my_ae)

aqp = Plot()
fpr_plot = aqp.plot_group_metric(xtab, 'fpr')

from aequitas.bias import Bias

b = Bias()
bdf = b.get_disparity_predefined_groups(xtab, 
                    original_df=my_ae, 
                    ref_groups_dict={"Gender":'Male', 'Age_group':'Age more than 74'}, 
                    alpha=0.05, 
                    check_significance=False)
fpr_disparity = aqp.plot_disparity(bdf, group_metric='fpr_disparity', 
                                   attribute_name='Gender')

p = aqp.plot_group_metric_all(xtab, metrics=['ppr','pprev','fnr','fpr'], ncols=4)

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










