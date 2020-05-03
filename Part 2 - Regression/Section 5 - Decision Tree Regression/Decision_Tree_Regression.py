# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 15:00:01 2020

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
x=a.iloc[:,1:2]  #Tool:make sure this is a matrix
y=a.iloc[:,-1]   #Target:make sure this is a Vector

'''
since the input data is very small, no need of splitting the data into 
testing and training
'''

#Fitting the data to Decision Tree Module:
from sklearn.tree import DecisionTreeRegressor
reg=DecisionTreeRegressor(random_state=0)
reg.fit(x,y)
#predicting the data
y_predict=reg.predict(x)

#Plotting the Graph:
'''
for higher resolution of plots we need to split the 
plots to minute level i.e 0.1 or 0.01 
'''
#Higher resolution Graph Setting :
x_min=x.min().min() # since x is data frame x.min gives a series .min will give value
x_max=x.max().max()
x_grid=np.arange(x_min,x_max,0.01)
x_grid=x_grid.reshape(len(x_grid),1)
y_predict=reg.predict(x_grid)
#plotting of Graph
plt.scatter(x,y,c='red')
plt.plot(x_grid,y_predict,c='blue') # y predict should contain X grid
plt.title('Decision Tree Module')
plt.xlabel('Exp')
plt.ylabel('Salary')
plt.show()

#Final Prediction Check:
y_predict_final=reg.predict([[6.5]])
print('Salary predicted as per Decision Tree model is : {:.2f}'.format(y_predict_final[0]))

