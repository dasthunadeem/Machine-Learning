# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 13:02:06 2020

@author: dasth
"""
#Importing Libraries and Data Set:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

a=pd.read_csv('Position_Salaries.csv')    

#Checking for Null and Categorical Variable:
b=a.isnull().sum()
'''
c=pd.value_counts(a['Country'])  #Categorical Variable
'''

#Splitting the Data set to Training and Testing Data:
x=a.iloc[:,1:2]  #Tool:make sure this is a matrix
y=a.iloc[:,-1]   #Target:make sure this is a Vector

'''
since the input data is very small, no need of splitting the data into 
testing and training
'''

"""
compare the result of Polinomial with linear 
"""
#For_Reference we creat linear regression
#Fitting the data to Linear Reg Module:
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x,y)
#predicting the data
y_pred=reg.predict(x)

#Fitting the data to Polinomial Reg Module:
#Converting X tool as a polinomial equation
from sklearn.preprocessing import PolynomialFeatures
reg_poly=PolynomialFeatures(degree=4)
# Transform is used conver it into polinomial equation
x_poly=reg_poly.fit_transform(x) 
reg_2=LinearRegression()
reg_2.fit(x_poly, y)
#predicting the data
y_pred_poly=reg_2.predict(reg_poly.fit_transform(x))

#Plotting the Graph:
#Plotting of Linear reg:
plt.scatter(x,y,c='red')
plt.plot(x,y_pred)
plt.title('Linear regression plot')
plt.xlabel('Levels')
plt.ylabel('Salary')  
plt.show()
#plotting of Polinomial Regression:
plt.scatter(x,y,c='red')
plt.plot(x,y_pred_poly,c='black')
plt.title('Polinomial regression plot')
plt.xlabel('Levels')
plt.ylabel('Salary')  
plt.show()

#Final Prediction Check: 
#Predicting the salary of Employee with 6.5 exp in Linear Reg:
pred_lin=reg.predict([[6.5]])
print('salary predicted as linear model is : {:.2f}'.format(pred_lin[0]))

#Predicting the salary of Employee with 6.5 exp in Polinomial Reg:
pred_poly=reg_2.predict(reg_poly.fit_transform([[6.5]]))
print('salary predicted as Polinomial model is : {:.2f}'.format(pred_poly[0]))






