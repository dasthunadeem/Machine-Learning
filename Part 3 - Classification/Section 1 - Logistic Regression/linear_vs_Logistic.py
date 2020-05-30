# -*- coding: utf-8 -*-
"""
Created on Fri May 29 20:50:53 2020

@author: dasth
"""
'''
intention is to derive why we cannot use Linear Regression
in classification Problem
'''

'''
Peolple having plasma more than 140 are said to be having Diabeties
'''
import matplotlib.pyplot as plt
from pandas import DataFrame as df

data = [[100, 0],[80,0],[75,0],[60,0],
        [150, 1],[165,1],[180,1],[170,1]]
a=df(data,columns=['plasmascore','diabetes'])
x=a.iloc[:,:-1]
y=a.iloc[:,-1]

from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x,y)

plt.scatter(x, y)
plt.plot(x,reg.predict(x))
plt.title('Linear Model Without Outliers')
plt.show()

'''
This Score of 125 is ok as per Standard 
'''
plasma=125
Threshold=reg.predict([[plasma]])
print('the Plasma score @ threshold 0.5 is {}'.format(plasma))

#Introdicing outlier for the Data Set
b=df({'plasmascore':[450,500],'diabetes':[1,1]})
a_new=a.append(b,ignore_index=True)
x_new=a_new.iloc[:,:-1]
y_new=a_new.iloc[:,-1]

reg.fit(x_new, y_new)

plt.scatter(x_new, y_new)
plt.plot(x_new,reg.predict(x_new))
plt.title('Linear Model With Outliers')
plt.show()

plasma_new=160
Threshold_new=reg.predict([[plasma_new]])
print('the Plasma score @ threshold 0.5 is {}'.format(plasma_new))

