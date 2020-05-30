# -*- coding: utf-8 -*-
"""
Created on Fri May 29 18:23:32 2020

@author: dasth
"""
'''
Ridge and Lasoregression are used 
to reduce the overfitting in Linear Regression 
'''
from sklearn.datasets import load_boston
import numpy as np
from pandas import DataFrame as df
import  matplotlib.pyplot as plt
import seaborn as sns

#Exploratory Data ANalysis:
data=load_boston()

a=np.concatenate((data['data'],data['target'].reshape(-1,1)),axis=1)

data['feature_names']

a=df(a,columns=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX',
                'RM', 'AGE', 'DIS', 'RAD','TAX', 'PTRATIO', 'B',
                'LSTAT','Price'])

sns.heatmap(a.isnull(),yticklabels=False,cbar=False)
plt.show()

x=a.iloc[:,:-1]
y=a.iloc[:,-1]


#Linear Regression with K - Cross validation:
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
reg=LinearRegression()
k_cross_linear=cross_val_score(reg, x,y,
                        scoring='neg_mean_squared_error',
                        cv=5, n_jobs=-1)
mean_linear=k_cross_linear.mean()
# mean squared error should be around 0 for better rigression model

#Ridge Regression
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
reg_ridge=Ridge()
param={'alpha':[1e-15,1e-10,1e-5,1e-1,51,5,10,15,20,35,50.65,100]}
grid_search_ridge=GridSearchCV(estimator=reg_ridge, param_grid=param,
                         scoring='neg_mean_squared_error',
                         cv=5, n_jobs=-1)
grid_search_ridge=grid_search_ridge.fit(x,y)

best_pram_ridge=grid_search_ridge.best_params_
mean_ridge_ridge=grid_search_ridge.best_score_


#Lasso Regression
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
reg_lasso=Lasso()
param_lasso={'alpha':[1e-15,1e-10,1e-5,1e-1,51,
                      5,10,15,20,35,50.65,100]}
grid_search_lasso=GridSearchCV(estimator=reg_lasso,
                               param_grid=param_lasso,
                               scoring='neg_mean_squared_error',
                               cv=5, n_jobs=-1)
grid_search_lasso=grid_search_lasso.fit(x,y)

best_pram_lasso=grid_search_lasso.best_params_
mean_lasso=grid_search_lasso.best_score_

#Predicting With Ridge and Lasso:
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.7,
                                               random_state=0)
predict_ridge=grid_search_ridge.predict(x_test)
predict_lasso=grid_search_lasso.predict(x_test)

sns.distplot(y_test-predict_ridge)
plt.title('Ridge Model')
plt.show()

sns.distplot(y_test-predict_lasso)
plt.title('Lasso Model')
plt.show()



