# -*- coding: utf-8 -*-
"""
Created on Fri Oct 06 19:29:42 2017

@author: Admin
"""

import seaborn as sns
import pandas
from matplotlib import pyplot as plt
from sklearn import svm
import numpy as np
from sklearn.metrics import accuracy_score

X_train = pandas.read_csv("C:\Users\nisha\Desktop\ML\\train.csv")
X_test = pandas.read_csv("C:\Users\nisha\Desktop\ML\\test.csv")
#%%
Y_test = X_test.iloc[:,55]
X_test = X_test.iloc[:,1:55]
Y_train = X_train.iloc[:,55] #1
X_train = X_train.iloc[:,1:55] #55 
print X_train.shape , X_test.shape

#%%
## not removing outliers as mean and std is nan
"""
cols=X_train.columns
print(cols)
print X_train.shape
for col in cols:
    mean=X_train[col].mean()
    print mean
    std=X_train[col].std()
    print std
    X_train=X_train[X_train[col]>mean+2*std]

print X_train.shape
"""
#%%
clf=svm.SVC(kernel='rbf', C=1000, gamma=1e-06)
clf.fit(X_train, Y_train)
pred=clf.predict(X_test)
print(accuracy_score(pred, Y_test))
     
#%%

size = 10 

#create a dataframe with only 'size' features
data=X_train.iloc[:,:size]  #10

#get the names of all the columns
cols=data.columns 

# Calculates pearson co-efficient for all combinations
data_corr = data.corr()
#print('correlation')
#print(data_corr)
# Set the threshold to select only only highly correlated attributes
threshold = 0.5

# List of pairs along with correlation above threshold
corr_list = []

#Search for the highly correlated pairs
for i in range(0,size): #for 'size' features
    for j in range(i+1,size): #avoid repetition
        if (data_corr.iloc[i,j] >= threshold and data_corr.iloc[i,j] < 1) or (data_corr.iloc[i,j] < 0 and data_corr.iloc[i,j] <= -threshold):
            corr_list.append([data_corr.iloc[i,j],i,j]) #store correlation and columns index
"""
#all the pairs
for i in range(0,size): #for 'size' features
    for j in range(i+1,size): #avoid repetition
        corr_list.append([data_corr.iloc[i,j],i,j]) #store correlation and columns index
"""
#Sort to show higher ones first            
s_corr_list = sorted(corr_list,key=lambda x: -abs(x[0]))
#Print correlations and column names
for v,i,j in s_corr_list:
    print ("%s and %s = %.2f" % (cols[i],cols[j],v))

#%%
import matplotlib.patches as mpatches
fig=plt.figure()
k=1
for v,i,j in s_corr_list:
    #sns.pairplot(X_train, hue="Cover_Type", size=6, x_vars=cols[i],y_vars=cols[j] )
    red_patch=mpatches.Patch(color='red', label=cols[i])
    blue_patch=mpatches.Patch(color='blue', label=cols[j])
    plt.scatter(X_train.iloc[:,i], X_train.iloc[:,j], c=Y_train, s=75, cmap=plt.cm.Oranges)
    plt.legend(handles=[red_patch, blue_patch])
    fig.savefig("D:\IIITD\MachineLearning\Project\Plots\%s.png"%k)
    k+=1
    plt.show() 

   
#%%
#adding new features

#print(dataset)
import math
from sklearn.model_selection import train_test_split
#print(data['_Distance_To_Hydrology'])
X_train['distance_to_Hydrology']=X_train.apply(lambda row: math.sqrt(math.pow(row['Horizontal_Distance_To_Hydrology'],2)+math.pow(row['Vertical_Distance_To_Hydrology'],2)), axis=1)
X_test['distance_to_Hydrology']=X_test.apply(lambda row: math.sqrt(math.pow(float(row['Horizontal_Distance_To_Hydrology']),2)+math.pow(float(row['Vertical_Distance_To_Hydrology']),2)), axis=1)
#%%
X_train.distance_to_Hydrology=X_train.distance_to_Hydrology.map(lambda x: 0 if np.isinf(x).any() else x)
X_test.distance_to_Hydrology=X_test.distance_to_Hydrology.map(lambda x: 0 if np.isinf(x).any() else x)
X_train.distance_to_Hydrology=X_train.distance_to_Hydrology.map(lambda x: 0 if np.isnan(x).any() else x)
X_test.distance_to_Hydrology=X_test.distance_to_Hydrology.map(lambda x: 0 if np.isnan(x).any() else x)
print(X_train.columns)
#%%

#drop irrelevant columns #2
pandas.set_option('display.max_columns', None)
#print(dataset.describe())
#print(X_train.columns)
for c in X_train.columns:
    if X_train[c].std()==0:
        X_train.drop(c,axis=1,inplace=True)
        X_test.drop(c, axis=1, inplace=True)
        print('drop', c)
"""
v, i, j=max(s_corr_list)
print(v, cols[i], cols[j])
X_train.drop(cols[i], axis=1, inplace=True) #total columns dropped =3
X_test.drop(cols[i], axis=1, inplace=True)
"""
print('train', X_train.shape)
print('test', X_test.shape)
#print(dataset.)
#%%
count=0
for v in X_test['distance_to_Hydrology']:
    if np.isinf(v):
        count+=1
print(count)
#print(X_test['distance_to_Hydrology'])
print('train', X_train.shape)
print('test', X_test.shape)
#%%
"""
X_train.drop('Horizontal_Distance_To_Hydrology', axis=1, inplace=True)
X_train.drop('Vertical_Distance_To_Hydrology', axis=1, inplace=True)
X_test.drop('Horizontal_Distance_To_Hydrology', axis=1, inplace=True)
X_test.drop('Vertical_Distance_To_Hydrology', axis=1, inplace=True)
"""
print cols
#%%
# elevation vs horz dist to hydrology
plt.scatter(X_train.iloc[:,0], X_train.iloc[:,3], c=Y_train, s=6, cmap=plt.cm.Oranges)
plt.show()

#%%
X_train['elev_HD_hydrology']=X_train.apply(lambda row: row['Elevation']-0.2*row['Horizontal_Distance_To_Hydrology'], axis=1)
X_test['elev_HD_hydrology']=X_test.apply(lambda row: row['Elevation']-0.2*row['Horizontal_Distance_To_Hydrology'], axis=1)

#%%
#elevation-0.2*HD vs HD
plt.scatter(X_train.iloc[:,0]-0.2*X_train.iloc[:, 3], X_train.iloc[:,3], c=Y_train, s=6, cmap=plt.cm.Oranges)
plt.show()

#%%
# elevation vs vert dist to hydrology
plt.scatter(X_train.iloc[:,0], X_train.iloc[:,4], c=Y_train, s=6, cmap=plt.cm.Oranges)
plt.show()
#%%
#elevation-VD vs VD
plt.scatter(X_train.iloc[:,0]-X_train.iloc[:, 4], X_train.iloc[:,4], c=Y_train, s=6, cmap=plt.cm.Oranges)
plt.show()

#%%

X_train['elev_VD_hydrology']=X_train.apply(lambda row: row['Elevation']-row['Vertical_Distance_To_Hydrology'], axis=1)
X_test['elev_VD_hydrology']=X_test.apply(lambda row: row['Elevation']-row['Vertical_Distance_To_Hydrology'], axis=1)

#%%
# elevation vs roadways
plt.scatter(X_train.iloc[:,0], X_train.iloc[:,5], c=Y_train, s=6, cmap=plt.cm.Oranges)
plt.show()

#%%
#add new feature for mean amenities
X_train['mean_distance_to_amenities']=X_train.apply(lambda row: (row['Horizontal_Distance_To_Hydrology']+row['Horizontal_Distance_To_Roadways']+row['Horizontal_Distance_To_Fire_Points'])/3.0, axis=1)
X_test['mean_distance_to_amenities']=X_test.apply(lambda row: (row['Horizontal_Distance_To_Hydrology']+row['Horizontal_Distance_To_Roadways']+row['Horizontal_Distance_To_Fire_Points'])/3.0, axis=1)

#%%

#fit and predict

clf=svm.SVC(kernel='rbf', C=1000, gamma=1e-06, decision_function_shape='ovr')
clf.fit(X_train, Y_train)
pred=clf.predict(X_test)
print(accuracy_score(pred, Y_test))

#%%
plt.scatter(X_train['Hillshade_9am'], X_train['Hillshade_3pm'], c=Y_train, s=75, cmap=plt.cm.Oranges)
plt.show()

#%%
#not yet used
print X_train['Aspect'].head()
X_train['Aspect2']=X_train.apply(lambda row: int(row['Aspect']-180) if row['Aspect']>180 else row['Aspect']+180, axis=1)
X_test['Aspect2']=X_test.apply(lambda row: int(row['Aspect']-180) if row['Aspect']>180 else row['Aspect']+180, axis=1)
X_train['Aspect2'].astype(int)
X_test['Aspect2'].astype(int)
print X_train.Aspect2.head(), X_test['Aspect2'].head()
#%%
X_train.drop('Aspect2', axis=1, inplace=True)
X_test.drop('Aspect2', axis=1, inplace=True)
#%%

X_train['VD_boolean']=X_train.apply(lambda row: 1 if row['Vertical_Distance_To_Hydrology'] >=0 else 0, axis=1)
X_test['VD_boolean']=X_test.apply(lambda row: 1 if row['Vertical_Distance_To_Hydrology'] >=0 else 0, axis=1)
#%%

X_train['sin_aspect']=X_train.apply(lambda row: math.sin(row['Aspect']), axis=1)
X_test['sin_aspect']=X_test.apply(lambda row: math.sin(row['Aspect']), axis=1)

#%%
print X_train.columns
#%%
#X_train[:,14:42].idxmax(1)
df=X_train.iloc[:, 14:52]
print df.shape
df.idxmax()

#%%

#X_train[,14:52] = lapply(coverData[,12:56], as.factor)
# Function to regroup dummy data in one column
wilderness=X_train.iloc[10:14]
"""
def reGroup(oldColumns, newLabels, columnName):
    for i in range(len(newLabels)):
        wilderness=wilderness[get(oldColumns[i])==1,paste(columnName):=newLabels[i]]

# Dummy old columns and new labels of Wilderness Area
newLabels <- c("Rawah","Neota","Comanche Peak","Cache la Poudre")
oldColumns <- c("Wilderness_Area1","Wilderness_Area2","Wilderness_Area3","Wilderness_Area4")
columnName <- "Wilderness_Area"

# Regrouping Wilderness Area
reGroup(oldColumns, newLabels, columnName)
"""
X_train.Wilderness_Area4.value_counts()
#%%
### Group one-hot encoded variables of a category into one single variable
cols = X_train.columns
r,c = X_train.shape
print cols
### Group one-hot encoded variables of a category into one single variable
cols = X_train.columns
r,c = X_train.shape

# Create a new dataframe with r rows, one column for each encoded category, and target in the end
new_data = pandas.DataFrame(index= np.arange(0,r), columns=['Wilderness_Area', 'Soil_Type', 'Cover_Type'])

# Make an entry in data for each r for category_id, target_value
for i in range(0,r):
    p = 0;
    q = 0;
    # Category1_range
    for j in range(10,14):
        if (X_train.iloc[i,j] == 1):
            p = j-9 # category_class
            break
    # Category2_range
    for k in range(14,54):
        if (X_train.iloc[i,k] == 1):
            q = k-13 # category_class
            break
    # Make an entry in data for each r
    new_data.iloc[i] = [p,q,X_train.iloc[i, c-1]]
    
# plot for category1
sns.countplot(x = 'Wilderness_Area', hue = 'Cover_Type', data = new_data)
plt.show()

# Plot for category2
plt.rc("figure", figsize = (25,10))
sns.countplot(x='Soil_Type', hue = 'Cover_Type', data= new_data)
plt.show()

