# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 20:27:27 2017

@author: nisha
"""

import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import GridSearchCV
from sklearn.learning_curve import validation_curve
from matplotlib import pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn import svm
from sklearn.model_selection import cross_val_score
import operator
import pickle



X_train=np.load('X_train_new.npy')
Y_train=np.load('Y_train.npy')
X_test=np.load('X_test_new.npy')
Y_test=np.load('Y_test_new.npy')

print X_train.shape

"""
param_grid={
'max_depth':[10,15,19,23,27,31,35],
'min_samples_split':[2,3,4,5],
'min_samples_leaf':[1,2,3,4]       
}
clf=GridSearchCV(DecisionTreeClassifier(random_state=1),param_grid,cv=5,verbose=5)
clf.fit(X_train,Y_train)
print "Grid Search on Decision Tree original data"
print clf.best_estimator_
print clf.best_score_
print clf.best_params_


param_grid={
'n_estimators':[100,200,300,400,500]    
}
base_model=DecisionTreeClassifier(random_state=1,max_depth=23)
clf=GridSearchCV(BaggingClassifier(random_state=1,base_estimator=base_model),param_grid,cv=5,verbose=5)
clf.fit(X_train,Y_train)
print "Grid Search on Bagged Original Data"
print clf.best_estimator_
print clf.best_score_
print clf.best_params_

param_grid={
'n_estimators':[100,200,300,400,500],
'min_samples_split':[2,3,4,5]   
}
clf=GridSearchCV(ExtraTreesClassifier(random_state=1),param_grid,cv=5,verbose=5)
clf.fit(X_train,Y_train)
print "Grid Search on Extra Tree original data set"
print clf.best_estimator_
print clf.best_score_
print clf.best_params_
param_grid={
'n_estimators':[100,200,300,400,500],
'learning_rate':[0.001,0.01,0.1,1,2,3]   
}

base_model=DecisionTreeClassifier(random_state=1,max_depth=23)
clf=GridSearchCV(AdaBoostClassifier(random_state=1,base_estimator=base_model),param_grid,cv=5,verbose=5)
clf.fit(X_train,Y_train)
print "Grid Search on  AdaBoost original data"
print clf.best_estimator_
print clf.best_score_
print clf.best_params_

base_model=DecisionTreeClassifier(random_state=1,max_depth=23,min_samples_leaf=1,min_samples_split=2,criterion='entropy')
clf=BaggingClassifier(base_estimator=base_model,n_estimators=500,random_state=1)
print "here2"
clf.fit(X_train,Y_train)
print   "Cross Validation score " , cross_val_score(clf,X_train,Y_train,cv=5).mean()
print "here3"
pred=clf.predict(X_train)

#print  "Classification Accuracy with BDT wth original data" ,  accuracy_score(pred,Y_test)
print   "Training Score " , accuracy_score(pred,Y_train)



clf=DecisionTreeClassifier(max_depth=23,random_state=1,min_samples_leaf=1,min_samples_split=2,criterion='entropy')
print "here2"
clf.fit(X_train,Y_train)
print   "Cross Validation score " , cross_val_score(clf,X_train,Y_train,cv=5).mean()
print "here3"
pred=clf.predict(X_train)
#print  "Classification Accuracy with DT wth original data" ,  accuracy_score(pred,Y_test)
print   "Training Score " , accuracy_score(pred,Y_train)

clf=ExtraTreesClassifier(n_estimators=500,random_state=1,min_samples_split=2,criterion='entropy')
print "here2"
clf.fit(X_train,Y_train)
print   "Cross Validation score " , cross_val_score(clf,X_train,Y_train,cv=5).mean()
l=clf.feature_importances_
res={}
for i in range(len(l)):
    res[i]=l[i]
print res
tt=[]
c=0
for k,v in sorted(res.items(),key=operator.itemgetter(1),reverse=True):
    tt.append(k)
    c+=1
    if(c==20):
        break

print tt    
#print clf.feature_importances
print "here3"
pred=clf.predict(X_train)
#print  "Classification Accuracy with ET wth original data" ,  accuracy_score(pred,Y_test)
print   "Training Score " , accuracy_score(pred,Y_train)


base_model=DecisionTreeClassifier(random_state=1,max_depth=23,min_samples_leaf=1,min_samples_split=2)
clf=AdaBoostClassifier(base_estimator=base_model,n_estimators=500,random_state=1,learning_rate=1)
print "here2"
clf.fit(X_train,Y_train)
print   "Cross Validation score " , cross_val_score(clf,X_train,Y_train,cv=5).mean()
print "here3"
pred=clf.predict(X_train)
#print  "Classification Accuracy with  wth original data" ,  accuracy_score(pred,Y_test)
print   "Training Score " , accuracy_score(pred,Y_train)


## validation curve 
base_model=DecisionTreeClassifier(random_state=1,max_depth=23,min_samples_leaf=1,min_samples_split=2)
clf=BaggingClassifier(base_estimator=base_model,random_state=1)
lst=np.arange(100,600,100)
ts,cv=validation_curve(clf,X_train,Y_train,param_name='n_estimators',param_range=lst)
plt.xlabel('n_estimators')
plt.ylabel('Mean CV score')
plt.plot(lst,np.mean(cv,axis=1),label="Cross Validation Score" , color="r")

## random Forest

clf=RandomForestClassifier(n_estimators=500,random_state=1,min_samples_split=2,criterion='entropy')
clf.fit(X_train,Y_train)
#print   "Cross Validation score " , cross_val_score(clf,X_train,Y_train,cv=5).mean()
pred=clf.predict(X_test)
print  "Classification Accuracy with RF wth original data" ,  accuracy_score(pred,Y_test)
#print   "Training Score " , accuracy_score(pred,Y_train)
with open('Best_RandomForest.model','wb') as f1:
    pickle.dump(clf,f1)   ## all features
print "model saved"



clf=svm.SVC(kernel='rbf',C=1000,gamma=1e-06,decision_function_shape='ovr')
clf.fit(X_train,Y_train)
pred=clf.predict(X_test)
print "Classification Accuracy with SVM wth original data" , accuracy_score(pred,Y_test)
"""
clf=ExtraTreesClassifier(n_estimators=500,random_state=1,min_samples_split=2,criterion='entropy')
print "here2"
clf.fit(X_train,Y_train)
pred=clf.predict(X_test)
print  "Classification Accuracy with ET wth original data" ,  accuracy_score(pred,Y_test)

with open('Best_ExtraTree.model','wb') as f1:
    pickle.dump(clf,f1)
print "model saved"  ## all features  
 