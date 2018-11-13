# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 17:53:51 2017

@author: nisha
"""

import pickle
from sklearn.metrics import roc_curve
import numpy as np
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.ensemble import ExtraTreesClassifier

X_train=np.load('X_train_new.npy')
Y_train=np.load('Y_train.npy')
X_test=np.load('X_test_new.npy')
Y_test=np.load('Y_test_new.npy')
print X_train.shape
Y_train = label_binarize(Y_train, classes=[1,2,3,4,5,6,7])
Y_test = label_binarize(Y_test, classes=[1,2,3,4,5,6,7])
n_classes = Y_train.shape[1]
clf=OneVsRestClassifier(ExtraTreesClassifier(n_estimators=500,random_state=1,min_samples_split=2,criterion='entropy'))
y_score = clf.fit(X_train, Y_train).predict(X_test)

fpr = dict()
tpr = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(Y_test[:, i], y_score[:, i])
    
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
plt.title("Roc Curve for Extra Tree Classifier")
plt.xlabel("False Postive Rate")
plt.ylabel("True Postive Rate")
plt.plot(fpr["macro"], tpr["macro"],linewidth=2,color='darkorange') 
plt.show()  
"""    
lw=2    
plt.title("Roc Curve for Extra Tree Classifier")
plt.xlabel("False Postive Rate")
plt.ylabel("True Postive Rate")
plt.plot(fpr[2], tpr[2], color='darkorange',lw=lw)
plt.show()
"""
#****************************************************************************************************************


##******************************************************************************************************************
      
   





    
    
    