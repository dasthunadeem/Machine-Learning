# -*- coding: utf-8 -*-
"""
Created on Sat May 30 21:25:47 2020

@author: dasth
"""
#Importing Libraries and Data Set:

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


a=pd.read_csv('Churn_Modelling.csv',usecols=['CreditScore', 'Geography',
       'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
       'IsActiveMember', 'EstimatedSalary', 'Exited'])

#Checking for Null Value:
plt.figure(figsize=(10,5))
sns.heatmap(a.isnull(),yticklabels=False,cbar=False)
plt.show()

#checking for Categorical Variable:
plt.figure(figsize=(10,5))
sns.countplot('Geography',hue='Gender',data=a)
plt.show()

#Checking for Imbalance DataSet:
plt.figure(figsize=(10,5))
sns.countplot('Exited',data=a)
plt.show()

#Encoding Categorical Variable:
from sklearn.preprocessing import LabelEncoder
encode=LabelEncoder()
a['Gender']=encode.fit_transform(a['Gender'])
a=pd.get_dummies(data=a,drop_first=True)
a=a[['CreditScore','Geography_Germany', 'Geography_Spain',
     'Gender', 'Age', 'Tenure',
    'Balance', 'NumOfProducts','HasCrCard',
    'IsActiveMember', 'EstimatedSalary', 'Exited']]

#Checking for Correlation:
plt.figure(figsize=(10,5))
sns.heatmap(a.corr(),annot=True)
plt.show()

#Splitting Data to Tool and Target:
x=a.iloc[:,:-1]
y=a.iloc[:,-1]

#Splitting Data to Train and Test Split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train, y_test = train_test_split(x,y,test_size=0.8,
                                                  random_state=0)
#Feature Scaling:
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test)

#fitiing data to XgBoost Classifier:
from xgboost import XGBClassifier
model=XGBClassifier()

#Hyperparameter optimization using RandomsearchCV:
from sklearn.model_selection import RandomizedSearchCV
param={'learning_rate'    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ],
       'max_depth'        : [ 3, 4, 5, 6, 8, 10, 12, 15],
       'min_child_weight' : [ 1, 3, 5, 7 ],
       'gamma'            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
       'colsample_bytree': [ 0.3, 0.4, 0.5 , 0.7 ]}
random_search=RandomizedSearchCV(estimator=model,
                                 param_distributions=param,
                                 scoring='accuracy',
                                 n_jobs=-1,
                                 cv=10)
random_search.fit(x_train, y_train)
estimator=random_search.best_estimator_
parameter=random_search.best_params_

#refitting Model on Hyperparameter:
model_new=XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.5, gamma=0.0,
              learning_rate=0.1, max_delta_step=0, max_depth=4,
              min_child_weight=3, missing=None, n_estimators=100, n_jobs=1,
              nthread=None, objective='binary:logistic', random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
              silent=None, subsample=1, verbosity=1)
model_new.fit(x_train, y_train)

#Checking for average accuracy with K_Cross validation:
from sklearn.model_selection import cross_val_score
K_Cross=cross_val_score(estimator=model_new, X=x_train, y=y_train,cv=10,n_jobs=-1)
accuracy=K_Cross.mean()
print('accuracy is {}'.format(accuracy))










