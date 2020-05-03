# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 18:21:57 2020

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

#Splitting the Data to Target and Tool:
x=a.iloc[:,1:2]
y=a.iloc[:,-1]

#Fitting the data to Random Forest Module:
from sklearn.ensemble import RandomForestRegressor
reg=RandomForestRegressor(n_estimators=300,random_state=0)
reg.fit(x,y)
#predicting the data
y_predict=reg.predict(x)

#Plotting the Graph:
"""
for higher resolution graph we create x_grid
"""
x_min=x.min().min()
x_max=x.max().max()
x_grid=np.arange(x_min,x_max,0.01)
x_grid=x_grid.reshape(len(x_grid),1)
y_predict_grid=reg.predict(x_grid)
plt.scatter(x,y,c='red')
plt.plot(x_grid,y_predict_grid)
plt.title('Random Forest Model')
plt.xlabel('Level')
plt.ylabel('Exp')
plt.show()

#Final Prediction Check:
y_predict_final=reg.predict([[6.5]])
print('Salary predicted as per Decision Tree model is : {:.2f}'.format(y_predict_final[0]))