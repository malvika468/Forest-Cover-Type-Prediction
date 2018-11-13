# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 19:27:55 2017

@author: nisha
"""
import numpy as np
from sklearn.metrics import accuracy_score
import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler

X_train=np.load('X_train_new.npy')
Y_train=np.load('Y_train.npy')
X_test=np.load('X_test_new.npy')
Y_test=np.load('Y_test_new.npy')
print  "Data shape" ,  X_train.shape , X_test.shape

with open('Best_ExtraTree.model','rb') as f1:
    clf=pickle.load(f1)
predictions=clf.predict(X_test) 
print  "Classification Accuracy with Extra Tree " ,  accuracy_score(predictions,Y_test)
    
##**************************************************************************************************
## load data


scaler=MinMaxScaler()
scaler.fit(X_train)
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.fit_transform(X_test)

with open('Best_MLP.model','rb') as f1:
    clf=pickle.load(f1)
print "Accuracy with MLP" ,  clf.score(X_test_scaled,Y_test)


##***********************************************************************************************************



with open('Best_RandomForest.model','rb') as f1:
    clf=pickle.load(f1)
predictions=clf.predict(X_test) 
print  "Classification Accuracy with Random Forest  " ,  accuracy_score(predictions,Y_test)


##*****************************************************************************************************


with open('Best_AdaBoost.model','rb') as f1:
    clf=pickle.load(f1)
predictions=clf.predict(X_test) 
print  "Classification Accuracy with AdaBoost  " ,  accuracy_score(predictions,Y_test)


#***********************************************************************************************************



with open('Best_BaggedDecisionTree.model','rb') as f1:
    clf=pickle.load(f1)
predictions=clf.predict(X_test) 
print  "Classification Accuracy with Bagging Decision Tree   " ,  accuracy_score(predictions,Y_test)


#**********************************************************************************************************





