# -*- coding: utf-8 -*-
"""
Created on Wed May  6 23:41:43 2020

@author: dasth
"""
#Importing Libraries and Data Set:
import pandas as pd 
a=pd.read_csv('Churn_Modelling.csv')

#Checking for Null and Categorical Variable:
b=a.isnull().sum()

#Splitting the Data to Target and Tool:
x=a.iloc[:,3:-1]
y=a.iloc[:,-1]

#Encoding Categorical Variable
from sklearn.preprocessing import LabelEncoder
encode=LabelEncoder()
x['Gender']=encode.fit_transform(x['Gender'])
x=pd.get_dummies(x,drop_first=True)

#Splitting the Data set to Training and Testing Data:
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=0)

#Fitting the data to XGB Module:
from xgboost import XGBClassifier
classifier=XGBClassifier()
classifier.fit(x_train, y_train)

#Prediction of Training Data:
y_predict_train=classifier.predict(x_train)

#Prediction of Testing Data:
y_predict_test=classifier.predict(x_test)

#Confusion Metrix to evaluate the prediction:
from sklearn.metrics import accuracy_score
accuracy_train=accuracy_score(y_train, y_predict_train)
accuracy_test=accuracy_score(y_test, y_predict_test)

#Applying K Fold Cross Validation:
from sklearn.model_selection import cross_val_score
cross=cross_val_score(estimator=classifier,X=x_train,y=y_train,
                      cv=10,n_jobs=-1)
accuracy_k_cross=cross.mean()

print('Accuracy of the model with XGB is {} %'.format(accuracy_test*100))
print('Accuracy of the model with K_Cross_Validation is {} %'.format(accuracy_k_cross*100))




