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

#Dummy Variables
x,y = np.split(data, (data.shape[1] - 1,), axis = 1)
df = pd.DataFrame(data = x)
df = pd.get_dummies(df, columns = [0,1], drop_first = True)

#MINMAX SCaler
x = np.array(df)
min_max_scaler = preprocessing.MinMaxScaler()
x = min_max_scaler.fit_transform(x)

data = np.c_[x, y]



# validate feature importance
model = RandomForestRegressor()
x,y = np.split(data, (data.shape[1] - 1,), axis=1)
model.fit(x, y.ravel())
importance = model.feature_importances_


d1 = d1.drop(["Healthy","Height","age_group","Gender","Ethnicity","Education","Smoke"], axis = 1)
a = d1.columns
a = pd.DataFrame(a)
a.columns = ["Attributes"]



for i in ["Taller than 175", "between 59~74", "Older 74", "Male", "Others Eth", "Edu=2", "Edu=3", "Edu=4","Edu=5","No-smoke"]:
    a = a.append({'Attributes': i}, ignore_index=True)




importance = pd.DataFrame(importance)
impo_df = pd.concat([a, importance], axis = 1)
impo_df.columns = ["Attributes", "Impo"]
print(impo_df)
i = impo_df.sort_values(by=['Impo'])



# balance classes
smo = SMOTE(random_state = 45)
x_smo, y_smo = smo.fit_resample(x, y)
data = np.c_[x_smo, y_smo]
# data = pd.DataFrame(data)
# print(Counter(data[64]))


#----------------------------------------------------
xtrain, xtest, ytrain, ytest = train_test_split(x_smo, y_smo, test_size=0.30, random_state=10)
from sklearn.svm import SVC
from sklearn import svm

time_start=time.time()

model_SVM = SVC()
model_SVM.fit(xtrain, ytrain)
pred_svm = model_SVM.predict(xtest)

time_end=time.time()
print('time cost',time_end-time_start,'s')

print(accuracy_score(pred_svm,ytest))
print(confusion_matrix(ytest,pred_svm))
print(classification_report(ytest,pred_svm))


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

from sklearn.linear_model import LogisticRegression

time_start=time.time()

model_LR = LogisticRegression()
model_LR.fit(xtrain, ytrain)
pred_lr = model_LR.predict(xtest)

time_end=time.time()
print('time cost',time_end-time_start,'s')


print(accuracy_score(pred_lr,ytest))
print(confusion_matrix(ytest,pred_lr))
print(classification_report(ytest,pred_lr))

from sklearn.naive_bayes import GaussianNB

time_start=time.time()

model_NB = GaussianNB()
model_NB.fit(xtrain, ytrain)
pred_nb = model_NB.predict(xtest)

time_end=time.time()
print('time cost',time_end-time_start,'s')

print(accuracy_score(pred_nb,ytest))
print(confusion_matrix(ytest, pred_nb))
print(classification_report(ytest, pred_nb))

from sklearn.ensemble import GradientBoostingClassifier

time_start=time.time()

model_GB = GradientBoostingClassifier()
model_GB.fit(xtrain, ytrain)
pred_gb = model_GB.predict(xtest)

time_end=time.time()
print('time cost',time_end-time_start,'s')

print(accuracy_score(pred_gb,ytest))
print(confusion_matrix(ytest, pred_gb))
print(classification_report(ytest, pred_gb))
#------------------------------------------------------------------------------


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
    
    
#---------------------------------------------------
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
model = RandomForestClassifier()
n_estimators = [10, 25, 50, 100, 150,200]
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
    

#---------------------------------------------------

model_LR = RandomForestClassifier(n_estimators=150, max_depth=None, min_samples_split=2,bootstrap=True)
model_LR.fit(xtrain, ytrain)
pred_lr = model_LR.predict(xtest)
print(accuracy_score(pred_lr,ytest))
plot_confusion_matrix(model_RF,
                      xtest,
                      ytest,
                      values_format = "d",
                      display_labels = ["Disease", "Healthy"])

#*************************************************************************
time_start=time.time()
score_RF = cross_val_score(RandomForestClassifier(n_estimators=50, max_depth=10, max_features="log2", min_samples_split=2,bootstrap=True),x_smo, y_smo, cv = 10)
print(np.average(score_RF))
time_end=time.time()
print('time cost',time_end-time_start,'s')
#*************************************************************************


#----------------------------------------------------
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.30, random_state=10)
clf = svm.SVC(C=512, kernel='rbf', gamma=0.01, decision_function_shape='ovr')
clf.fit(xtrain, ytrain.ravel())
score = clf.predict(xtest)
print(accuracy_score(score,ytest)) #0.692307
print(confusion_matrix(ytest, score))


X_train_prediction = clf.predict(xtrain)
training_data_accuracy = accuracy_score(ytrain, X_train_prediction)
print("Accuracy Score of training data: ", training_data_accuracy)

#accuracy score on training data
X_test_prediction = clf.predict(xtest)
testing_data_accuracy = accuracy_score(ytest, X_test_prediction)
print("Accuracy Score of test data: ", testing_data_accuracy)











score = pd.DataFrame(score)
score.columns = ["score"]


entity_id = {'entity_id':[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]}
entity_id = pd.DataFrame(entity_id)

data = pd.read_csv('/Users/downey/Desktop/v1.csv', sep=',')
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
print(zz)







