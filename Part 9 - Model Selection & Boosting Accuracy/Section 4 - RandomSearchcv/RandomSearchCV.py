# -*- coding: utf-8 -*-
"""
Created on Sat May 30 02:20:39 2020

@author: dasth
"""
#Importing Libraries and Data Set:
import pandas as pd 
import numpy as np

a=pd.read_csv('Social_Network_Ads.csv')
a=a.iloc[:,2:]

#Checking for Null and Categorical Variable:
b=a.isnull().sum()

'''
c=pd.value_counts(a['Gender'])
'''

#Splitting the Data to Target and Tool:
x=a.iloc[:,0:2]
y=a.iloc[:,-1]

#Splitting the Data set to Training and Testing Data:
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.75,
                                               random_state=0)

#Feature scaling:
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test)


#Fitting the data to  Random_Forest Module:
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=10,criterion="entropy",
                                  random_state=50 )

classifier.fit(x_train,y_train)

from sklearn.model_selection import cross_val_score
k_score_train=cross_val_score(estimator=classifier,
                              X=x_train, y=y_train,
                              scoring='accuracy',
                              cv=10, n_jobs=-1)
accuracy_train=k_score_train.mean()

k_score_test=cross_val_score(estimator=classifier,
                              X=x_test, y=y_test,
                              scoring='accuracy',
                              cv=10, n_jobs=-1)
accuracy_test=k_score_test.mean()

#Random Search CV:
from sklearn.model_selection import RandomizedSearchCV

param={'max_depth':[3,5,10,None],'n_estimators':[10,100,200,300,400,500],
       'max_features':np.arange(1,3),
       'criterion':['gini','entropy'],
       'bootstrap':[True,False],'min_samples_leaf':np.arange(1,4)}


model=RandomizedSearchCV(estimator=classifier,
                         param_distributions=param,
                         scoring='accuracy',
                         n_jobs=-1,
                         cv=10)

model.fit(x_train,y_train)

Best_Parameter=model.best_params_


#Rebuilding the Model after HyperParameter Optimisation:
classifier_opt=RandomForestClassifier(n_estimators=300,
                                  criterion="entropy",
                                  max_depth=5,
                                  max_features=1,
                                  max_leaf_nodes=2,
                                  bootstrap=True,)

classifier_opt.fit(x_train,y_train)


#Checking the Average Accuracy using K Cross Validation
k_score_train=cross_val_score(estimator=classifier_opt,
                              X=x_train, y=y_train,
                              scoring='accuracy',
                              cv=10, n_jobs=-1)
accuracy_train_cross=k_score_train.mean()

k_score_test=cross_val_score(estimator=classifier_opt,
                              X=x_test, y=y_test,
                              scoring='accuracy',
                              cv=10, n_jobs=-1)
accuracy_test_cross=k_score_test.mean()







