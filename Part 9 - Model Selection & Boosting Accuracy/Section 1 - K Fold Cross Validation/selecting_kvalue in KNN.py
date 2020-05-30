# -*- coding: utf-8 -*-
"""
Created on Sat May 30 00:06:01 2020

@author: dasth
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

a=pd.read_csv('k_value.csv',index_col=0)

x=a.iloc[:,:-1]
y=a.iloc[:,-1]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x, y,train_size=0.8,
                                               random_state=0)

from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

#Checking Precision and recall and accuracy for 5 neighbrs:
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,classification_report
knn_5=KNeighborsClassifier(n_neighbors=5)
knn_5.fit(x_train, y_train)
predict_5=knn_5.predict(x_test) 

accuracy_5=accuracy_score(y_test,predict_5)
report=classification_report(y_test,predict_5)


error_rate=[]
for x in range(1,40):
    knn=KNeighborsClassifier(n_neighbors=x)
    knn.fit(x_train, y_train)
    predict=knn.predict(x_test) 
    error_rate.append(np.mean(predict!=y_test))
    
plt.plot(range(1,40),error_rate,c='red')
plt.title('Selction of K value')
plt.xlabel('k_Value')
plt.ylabel('error_rate')
plt.show()

#Checking Precision and recall and accuracy for 20 neighbrs:
knn_20=KNeighborsClassifier(n_neighbors=20)
knn_20.fit(x_train, y_train)
predict_20=knn_20.predict(x_test) 

accuracy_20=accuracy_score(y_test,predict_20)
report_20=classification_report(y_test,predict_20)














