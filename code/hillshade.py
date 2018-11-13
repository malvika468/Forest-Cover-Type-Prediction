# -*- coding: utf-8 -*-
"""
Created on Sun Oct 08 23:09:43 2017

@author: Admin
"""
#import seaborn as sns
import pandas
from matplotlib import pyplot as plt
from sklearn import svm
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression

#%%
#read data
X_train = pandas.read_csv("D:\IIITD\MachineLearning\Project\\train.csv")
X_test = pandas.read_csv("D:\IIITD\MachineLearning\Project\\test.csv")
Y_test = X_test.iloc[:,55]
X_test = X_test.iloc[:,1:55]
Y_train = X_train.iloc[:,55] #1
X_train = X_train.iloc[:,1:55] #55 

#%%
#Y_train.Cover_Type.hist()
#print(Y_train.columns)
#print(X_train['Hillshade'])
print(Y_train.value_counts())
#%%

print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)


#%%
"""
#remove outliers from training (dont do it)
outlier=[]
data=X_train.iloc[:,:10] 
cols=data.columns
#print(cols)
for col in cols:
    outlier.append((data[col].mean()+2*data[col].std()))
#print(outlier)
for i in range(len(cols)):
    Y_train=Y_train[abs(X_train[cols[i]])<outlier[i]]
    X_train=X_train[abs(X_train[cols[i]])<outlier[i]]
X_train=X_train.reset_index(drop=True)
Y_train=Y_train.reset_index(drop=True)
print(X_train.head())
"""
#%%
#predictions for hillshade 3pm in training set
df_X=X_train[X_train['Hillshade_3pm'] !=0][['Slope', 'Aspect']]
df_Y=X_train[X_train['Hillshade_3pm'] !=0][['Hillshade_3pm']]
df_Xpred=X_train[X_train['Hillshade_3pm'] ==0][['Slope', 'Aspect']]
print(df_X.shape, df_Y.shape, df_Xpred.shape)
#%%
print 
#%%
clf=LinearRegression()
clf.fit(df_X, df_Y)
df_Ypred=clf.predict(df_Xpred)
print(len(df_Ypred))
j=0
for v in range(len(X_train.Hillshade_3pm)):
    if X_train.Hillshade_3pm[v]==0:
        X_train.Hillshade_3pm[v]=df_Ypred[j]
        j+=1
df_Y=X_train[X_train['Hillshade_3pm'] ==0]
print('train', df_Y.shape)

#%%
from sklearn.ensemble import GradientBoostingRegressor
clf=GradientBoostingRegressor()
clf.fit(df_X, df_Y)
df_Ypred=clf.predict(df_Xpred)
print(len(df_Ypred))
j=0
for v in range(len(X_train.Hillshade_3pm)):
    if X_train.Hillshade_3pm[v]==0:
        X_train.Hillshade_3pm[v]=df_Ypred[j]
        j+=1
df_Y=X_train[X_train['Hillshade_3pm'] ==0]
print('train', df_Y.shape)
#%%
print list(df_Ypred)
#%%
#predictions for hillshade 3pm in testing set
df_X=X_test[X_test['Hillshade_3pm'] !=0][['Slope', 'Aspect']]
df_Y=X_test[X_test['Hillshade_3pm'] !=0][['Hillshade_3pm']]
df_Xpred=X_test[X_test['Hillshade_3pm'] ==0][['Slope', 'Aspect']]
print(df_X.shape, df_Y.shape, df_Xpred.shape)

#%%

clf=LinearRegression()
clf.fit(df_X, df_Y)
df_Ypred=clf.predict(df_Xpred)

#print(df_Ypred)
j=0
#print(X_train.Hillshade_3pm)

for v in range(len(X_test.Hillshade_3pm)):
    if X_test.Hillshade_3pm[v]==0:
        X_test.Hillshade_3pm[v]=df_Ypred[j]
        j+=1
df_Y=X_test[X_test['Hillshade_3pm'] ==0]
print('test', df_Y.shape)

#%%
clf=GradientBoostingRegressor()
clf.fit(df_X, df_Y)
df_Ypred=clf.predict(df_Xpred)
print(len(df_Ypred))
j=0
for v in range(len(X_test.Hillshade_3pm)):
    if X_test.Hillshade_3pm[v]==0:
        X_test.Hillshade_3pm[v]=df_Ypred[j]
        j+=1
df_Y=X_test[X_test['Hillshade_3pm'] ==0]
print('test', df_Y.shape)
#%%
"""
clf=svm.SVC(kernel='rbf', C=1000, gamma=1e-06)
clf.fit(X_train, Y_train)
pred=clf.predict(X_test)
print(accuracy_score(pred, Y_test))
"""
#%%
#drop constant columns 
for c in X_train.columns:
    if X_train[c].std()==0:
        X_train.drop(c,axis=1,inplace=True)
        X_test.drop(c, axis=1, inplace=True)
        print('drop', c)

#%%

size = 10 
data=X_train.iloc[:,:size]  #10
cols=data.columns
data_corr = data.corr()
threshold = 0.5
corr_list = []

#Search for the highly correlated pairs
for i in range(0,size): #for 'size' features
    for j in range(i+1,size): #avoid repetition
        if (data_corr.iloc[i,j] >= threshold and data_corr.iloc[i,j] < 1) or (data_corr.iloc[i,j] < 0 and data_corr.iloc[i,j] <= -threshold):
            corr_list.append([data_corr.iloc[i,j],i,j]) #store correlation and columns index

s_corr_list = sorted(corr_list,key=lambda x: -abs(x[0]))
#Print correlations and column names
for v,i,j in s_corr_list:
    print ("%s and %s = %.2f" % (cols[i],cols[j],v))
    
"""
for v,i,j in s_corr_list:
    #sns.pairplot(X_train, hue="Cover_Type", size=6, x_vars=cols[i],y_vars=cols[j] )
    plt.scatter(X_train.iloc[:,i], X_train.iloc[:,j], c=Y_train, s=75, cmap=plt.cm.Oranges)
    plt.show() 
"""

#%%
#import matplotlib.patches as mpatches
#import seaborn as sns

k=1
for v,i,j in s_corr_list:
    #sns.pairplot(X_train, hue="Cover_Type", size=6, x_vars=cols[i],y_vars=cols[j] )
    #red_patch=mpatches.Patch(color='red', label=cols[i])
    #blue_patch=mpatches.Patch(color='blue', label=cols[j])
    fig=plt.figure()
    plt.xlabel(cols[i])
    plt.ylabel(cols[j])
    plt.scatter(X_train.iloc[:,i], X_train.iloc[:,j], c=Y_train, s=10, cmap=plt.cm.Oranges)
    #plt.legend(handles=[red_patch, blue_patch])
    fig.savefig("D:\IIITD\MachineLearning\Project\Plots\%s.png"%k)
    k+=1
    plt.show() 
 
#%%

v, i, j=max(s_corr_list)
print(v, cols[i], cols[j])
X_train.drop(cols[i], axis=1, inplace=True) #total columns dropped =3
X_test.drop(cols[i], axis=1, inplace=True)

#%%
import math

#adding new features

#print(data['_Distance_To_Hydrology'])
X_train['distance_to_Hydrology']=X_train.apply(lambda row: math.sqrt(math.pow(row['Horizontal_Distance_To_Hydrology'],2)+math.pow(row['Vertical_Distance_To_Hydrology'],2)), axis=1)
X_test['distance_to_Hydrology']=X_test.apply(lambda row: math.sqrt(math.pow(float(row['Horizontal_Distance_To_Hydrology']),2)+math.pow(float(row['Vertical_Distance_To_Hydrology']),2)), axis=1)

#%%
#elevation-0.2*HD
X_train['elev_HD_hydrology']=X_train.apply(lambda row: row['Elevation']-0.2*row['Horizontal_Distance_To_Hydrology'], axis=1)
X_test['elev_HD_hydrology']=X_test.apply(lambda row: row['Elevation']-0.2*row['Horizontal_Distance_To_Hydrology'], axis=1)

#%%
#elevation-VD
X_train['elev_VD_hydrology']=X_train.apply(lambda row: row['Elevation']-row['Vertical_Distance_To_Hydrology'], axis=1)
X_test['elev_VD_hydrology']=X_test.apply(lambda row: row['Elevation']-row['Vertical_Distance_To_Hydrology'], axis=1)

#%%
#add new feature for mean amenities
X_train['mean_distance_to_amenities']=X_train.apply(lambda row: (row['Horizontal_Distance_To_Hydrology']+row['Horizontal_Distance_To_Roadways']+row['Horizontal_Distance_To_Fire_Points'])/3.0, axis=1)
X_test['mean_distance_to_amenities']=X_test.apply(lambda row: (row['Horizontal_Distance_To_Hydrology']+row['Horizontal_Distance_To_Roadways']+row['Horizontal_Distance_To_Fire_Points'])/3.0, axis=1)

#%% 0.599
"""
#add hillshade 3pm * hillshade 9am
X_train['hillshade']=X_train.apply(lambda row: row['Hillshade_3pm']*row['Hillshade_9am'], axis=1)
X_test['hillshade']=X_test.apply(lambda row: row['Hillshade_3pm']*row['Hillshade_9am'], axis=1)
"""
#%% 
#add new feature for mean distance to fire and water
X_train['mean_distance_to_fire_water']=X_train.apply(lambda row: (row['Horizontal_Distance_To_Hydrology']+row['Horizontal_Distance_To_Fire_Points'])/2.0, axis=1)
X_test['mean_distance_to_fire_water']=X_test.apply(lambda row: (row['Horizontal_Distance_To_Hydrology']+row['Horizontal_Distance_To_Fire_Points'])/2.0, axis=1)

#%%
X_train['Is_water_source_below']=X_train.apply(lambda row: -1 if row['Vertical_Distance_To_Hydrology']<0 else 1, axis=1)
X_test['Is_water_source_below']=X_test.apply(lambda row: -1 if row['Vertical_Distance_To_Hydrology']<0 else 1, axis=1)

#%%
X_train['Aspect2']=X_train.apply(lambda row: (row['Aspect']+180)%360, axis=1)
X_test['Aspect2']=X_test.apply(lambda row: (row['Aspect']+180)%360, axis=1)

#%%

#add new feature for mean distance to fire and roadways
X_train['mean_distance_to_fire_road']=X_train.apply(lambda row: (row['Horizontal_Distance_To_Roadways']+row['Horizontal_Distance_To_Fire_Points'])/2.0, axis=1)
X_test['mean_distance_to_fire_road']=X_test.apply(lambda row: (row['Horizontal_Distance_To_Roadways']+row['Horizontal_Distance_To_Fire_Points'])/2.0, axis=1)


print(X_train.shape, Y_train.shape)
#print(X_test.shape, Y_test.shape, pred.shape)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.3)
print(X_train.shape, X_test.shape)
print(Y_train.shape, Y_test.shape)

#%%
X_train=X_train+X_test
Y_train=Y_train+Y_test
print X_train.shape, Y_train.shape
#%%
#testing accuracy score
clf=svm.SVC(kernel='rbf', C=1000, gamma=0.000001, decision_function_shape='ovr')
clf.fit(X_train, Y_train)
pred_test=clf.predict(X_test)
print(accuracy_score(pred_test, Y_test))
pred_train=clf.predict(X_train)
print(accuracy_score(pred_train, Y_train))

#%%
#training accuracy score
clf=svm.SVC(kernel='rbf', C=1000, gamma=0.000001, decision_function_shape='ovr')
clf.fit(X_train, Y_train)
pred=clf.predict(X_train)
print(accuracy_score(pred, Y_train))

#%%
#dont use
X_train['Aspect2']=X_train.apply(lambda row: int(row['Aspect']-180) if row['Aspect']>180 else row['Aspect']+180, axis=1)
X_test['Aspect2']=X_test.apply(lambda row: int(row['Aspect']-180) if row['Aspect']>180 else row['Aspect']+180, axis=1)
#%%
X_train.drop('Aspect2', axis=1, inplace=True)
X_test.drop('Aspect2', axis=1, inplace=True)
#%%
X_train['sin_aspect']=X_train.apply(lambda row: math.sin(row['Aspect']), axis=1)
X_test['sin_aspect']=X_test.apply(lambda row: math.sin(row['Aspect']), axis=1)
#%%
