# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 16:45:45 2020

@author: dasth
"""
#Importing Libraries and Data Set:
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

a=pd.read_csv('Salary_Data.csv')

#Checking for Null and Categorical Variable:
c=a.isnull().sum()  #check for null value
'''
c=pd.value_counts(a['Country'])  #Categorical Variable
'''

#Splitting the Data to Target and Tool:
x=a.iloc[:,0:1]  # Tool:make sure this is a matrix
y=a.iloc[:,-1]   #Target:make sure this is a Vector

#Splitting the Data set to Training and Testing Data:
from sklearn.model_selection import  train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.67,random_state=0)

#Fitting the data to Linear Reg Module:
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
#Predicting the data
y_predict_train=reg.predict(x_train)
y_predict_test=reg.predict(x_test)

##Plotting the Graph:
plt.scatter(x_train,y_train,c='red')
plt.plot(x_train,y_predict_train,c='green')
plt.title('Linear regression plot_Training Set')
plt.xlabel('Levels')
plt.ylabel('Salary')  
plt.show()
#plotting the testing grapgh :
plt.scatter(x_test,y_test,c='black')
plt.plot(x_train,y_predict_train,c='blue')
plt.title('Linear regression plot_Testing Set')
plt.xlabel('Levels')
plt.ylabel('Salary')  
plt.show()

#To find out mean squared error
from sklearn.metrics import mean_squared_error,r2_score,confusion_matrix
R=r2_score(y_test,y_predict_test)
mse_testing=mean_squared_error(y_test,y_predict_test)
print('MSE of testing data is {:.2f}'.format(mse_testing))
print('Rsquare score is {:.2f}'.format(R))

#Predicting for 6.5 Years of Expirience
y_predict_check=reg.predict([[6.5]])
print('Prediction of Salary of 6.5Years of Exp is Rs {:.2f}'.format(y_predict_check[0]))












