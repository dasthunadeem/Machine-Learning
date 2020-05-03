# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 18:52:31 2020

@author: dasth
"""
#Importing Libraries and Data Set:
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 

a=pd.read_csv('Position_Salaries.csv') #import data:

#Checking for Null and Categorical Variable:
b=a.isnull().sum()
'''
c=pd.value_counts(a['Country'])  #Categorical Variable
'''

#Splitting the Data to Target and Tool:
x=a.iloc[:,1:2] #Tool:make sure this is a matrix
y=a.iloc[:,2:3] #converted to matrix so that we can fit in Feature scaling

'''
since the input data is very small, no need of splitting the data into 
testing and training
'''

"""
svr model  does not support Feature Scaling 
hence we need to do it manually
"""
#Feature scaling:
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
sc_y=StandardScaler()
x=sc_x.fit_transform(x)
y=sc_y.fit_transform(y)
y=y.ravel() # converting it to a Vector

#Fitting the data to SVR Module:
from sklearn.svm import SVR
reg=SVR(kernel='rbf')
reg.fit(x,y)
#predicting the data
y_predict=reg.predict(x)
 
#Plotting the Graph:
plt.scatter(x,y,c='red')
plt.plot(x,y_predict,c='green')
plt.title('SVR plot')
plt.xlabel('Levels')
plt.ylabel('Salary')  
plt.show()

#Final Prediction Check: 
"""
1st we need to do the feature scaling of the input value 
i.e sc_trans, since fitting is already done in the model
2nd the resulted answer will be in the form of scaled value i.e bet -1 to 1
hence we need to inverse the output to get the correct estimation
"""
y_predict_final=sc_y.inverse_transform(reg.predict(sc_x.transform([[6.5]])))
print('Salary predicted as per Polinomial model is : {:.2f}'.format(y_predict_final[0]))









