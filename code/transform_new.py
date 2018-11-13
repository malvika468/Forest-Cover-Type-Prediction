# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 23:10:03 2017

@author: nisha
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 00:24:18 2017

@author: nisha
"""

import pandas
from matplotlib import pyplot as plt
from sklearn import svm
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
import math
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier

#%%
#read data
X_train = pandas.read_csv('C:\Users\\nisha\Desktop\ML\\train.csv')
X_test = pandas.read_csv('C:\Users\\nisha\Desktop\ML\\test.csv')
Y_test = X_test.iloc[:,55]
X_test = X_test.iloc[:,1:55]
Y_train = X_train.iloc[:,55] #1
X_train = X_train.iloc[:,1:55] #55 

print X_train.shape



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


df_X=X_test[X_test['Hillshade_3pm'] !=0][['Slope', 'Aspect']]
df_Y=X_test[X_test['Hillshade_3pm'] !=0][['Hillshade_3pm']]
df_Xpred=X_test[X_test['Hillshade_3pm'] ==0][['Slope', 'Aspect']]
print(df_X.shape, df_Y.shape, df_Xpred.shape)

#%%

clf=LinearRegression()
clf.fit(df_X, df_Y)
df_Ypred=clf.predict(df_Xpred)
j=0
for v in range(len(X_test.Hillshade_3pm)):
    if X_test.Hillshade_3pm[v]==0:
        X_test.Hillshade_3pm[v]=df_Ypred[j]
        j+=1
df_Y=X_test[X_test['Hillshade_3pm'] ==0]
print('test', df_Y.shape)



for c in X_train.columns:
    if X_train[c].std()==0:
        X_train.drop(c,axis=1,inplace=True)
        X_test.drop(c, axis=1, inplace=True)
        print('drop', c)



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

#%% 
#add new feature for mean distance to fire and water
X_train['mean_distance_to_fire_water']=X_train.apply(lambda row: (row['Horizontal_Distance_To_Hydrology']+row['Horizontal_Distance_To_Fire_Points'])/2.0, axis=1)
X_test['mean_distance_to_fire_water']=X_test.apply(lambda row: (row['Horizontal_Distance_To_Hydrology']+row['Horizontal_Distance_To_Fire_Points'])/2.0, axis=1)



X_train['Is_water_source_below']=X_train.apply(lambda row: -1 if row['Vertical_Distance_To_Hydrology']<0 else 1, axis=1)
X_test['Is_water_source_below']=X_test.apply(lambda row: -1 if row['Vertical_Distance_To_Hydrology']<0 else 1, axis=1)


X_train['sq_distance_to_road_water']=X_train.apply(lambda row: math.pow((row['Horizontal_Distance_To_Hydrology']-row['Horizontal_Distance_To_Roadways']),2) ,axis=1)
X_test['sq_distance_to_road_water']=X_test.apply(lambda row: math.pow((row['Horizontal_Distance_To_Hydrology']-row['Horizontal_Distance_To_Roadways']),2) , axis=1)


#####adddedd justt nowww
"""

X_train['mean_diff_to_fire_water']=X_train.apply(lambda row: (row['Horizontal_Distance_To_Hydrology']-row['Horizontal_Distance_To_Fire_Points']), axis=1)
X_test['mean_diff_to_fire_water']=X_test.apply(lambda row: (row['Horizontal_Distance_To_Hydrology']-row['Horizontal_Distance_To_Fire_Points']), axis=1)


X_train['Aspect2']=X_train.apply(lambda row: (row['Aspect']+180)%360, axis=1)
X_test['Aspect2']=X_test.apply(lambda row: (row['Aspect']+180)%360, axis=1)



X_train['sq_diff_to_fire_water']=X_train.apply(lambda row: math.pow(row['mean_diff_to_fire_water'] , 2) , axis=1)
X_test['sq_diff_to_fire_water']=X_test.apply(lambda row: math.pow(row['mean_diff_to_fire_water'],2), axis=1)


X_train['sq_dist_to_fire_water']=X_train.apply(lambda row: math.pow(row['mean_distance_to_fire_water'] , 2) , axis=1)
X_test['sq_dist_to_fire_water']=X_test.apply(lambda row: math.pow(row['mean_distance_to_fire_water'],2), axis=1)

##########################


#add new feature for mean distance to fire and road
X_train['mean_distance_to_fire_road']=X_train.apply(lambda row: (row['Horizontal_Distance_To_Roadways']+row['Horizontal_Distance_To_Fire_Points'])/2.0, axis=1)
X_test['mean_distance_to_fire_road']=X_test.apply(lambda row: (row['Horizontal_Distance_To_Roadways']+row['Horizontal_Distance_To_Fire_Points'])/2.0, axis=1)



#add new feature for diff   distance to road and water
X_train['mean_diff_to_road_water']=X_train.apply(lambda row: (row['Horizontal_Distance_To_Hydrology']-row['Horizontal_Distance_To_Roadways']) , axis=1)
X_test['mean_diff_to_road_water']=X_test.apply(lambda row: (row['Horizontal_Distance_To_Hydrology']-row['Horizontal_Distance_To_Roadways']) , axis=1)


X_train['sq_distance_to_water_road']=X_train.apply(lambda row: math.pow(row['mean_diff_to_road_water'] , 2) , axis=1)
X_test['sq_distance_to_water_road']=X_test.apply(lambda row: math.pow(row['mean_diff_to_road_water'],2), axis=1)

X_train['mean_diff_to_road_fire']=X_train.apply(lambda row: (row['Horizontal_Distance_To_Fire_Points']-row['Horizontal_Distance_To_Roadways']) , axis=1)
X_test['mean_diff_to_road_fire']=X_test.apply(lambda row: (row['Horizontal_Distance_To_Fire_Points']-row['Horizontal_Distance_To_Roadways']) , axis=1)


X_train['sq_diff_to_fire_road']=X_train.apply(lambda row: math.pow(row['mean_diff_to_road_fire'] , 2) , axis=1)
X_test['sq_diff_to_fire_road']=X_test.apply(lambda row: math.pow(row['mean_diff_to_road_fire'],2), axis=1)

#add new feature for squared distance to fire and road

X_train['sq_distance_to_fire_road']=X_train.apply(lambda row: math.pow(row['mean_distance_to_fire_road'] , 2) , axis=1)
X_test['sq_distance_to_fire_road']=X_test.apply(lambda row: math.pow(row['mean_distance_to_fire_road'],2), axis=1)

"""
#%%



#%%

np.save('X_train_MLP',X_train)
np.save('X_test_MLP',X_test)
