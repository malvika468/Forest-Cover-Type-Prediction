# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 19:12:13 2017

@author: nisha
"""


from matplotlib import pyplot as plt
from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
import pandas,pickle 
from sklearn.externals import joblib
import random
from sklearn.learning_curve import validation_curve
from sklearn.model_selection import cross_val_score

X_train=np.load('X_train_MLP.npy')
Y_train=np.load('Y_train.npy')
X_test=np.load('X_test_MLP.npy')
Y_test=np.load('Y_test_new.npy')
print X_train.shape

scaler=MinMaxScaler()
scaler.fit(X_train)
X_train_scaled=scaler.fit_transform(X_train)
print "here"
X_test_scaled=scaler.fit_transform(X_test)
print X_train_scaled.shape , Y_train.shape

"""
param_grid={
'hidden_layer_sizes': [25,37,50,80,100,120,150],
'alpha': [1,0.1,0.01,0.001,0.0001],
'activation': ["logistic", "relu", "tanh"],
'learning_rate_init':[0.1,1,0.001,3,5,0.05],
'solver':["lbfgs","sgd"]
}
clf=GridSearchCV(MLPClassifier(random_state=1),param_grid,cv=5,verbose=5)
print "Grid Search transformed data NN"
clf.fit(X_train_scaled,Y_train)
print clf.best_params_
print clf.best_score_
print clf.best_estimator_


clf=MLPClassifier(hidden_layer_sizes=100,activation='relu',solver='sgd',learning_rate_init=0.1,random_state=1,alpha=0.001,max_iter=700)
clf.fit(X_train_scaled,Y_train)
print   "Cross Validation score " , cross_val_score(clf,X_train_scaled,Y_train,cv=5).mean()
print "Accuracy with MLP" ,  clf.score(X_test_scaled,Y_test)
print   "Training Score " , clf.score(X_train_scaled,Y_train)

#with open('Best_MLP.model','wb') as f1:
    #pickle.dump(clf,f1)
"""
with open('Best_MLP.model','rb') as f1:
    clf=pickle.load(f1)


print   "Cross Validation score " , cross_val_score(clf,X_train_scaled,Y_train,cv=5).mean()
print "Accuracy with MLP" ,  clf.score(X_test_scaled,Y_test)
print   "Training Score " , clf.score(X_train_scaled,Y_train)
"""
lst=np.arange(20,120,20)
ts,cv=validation_curve(clf,X_train_scaled,Y_train,param_name='hidden_layer_sizes',param_range=lst)
plt.xlabel('Hidden Layers')
plt.ylabel('Mean CV score')
plt.plot(lst,np.mean(cv,axis=1),label="Cross Validation Score" , color="r")

"""








