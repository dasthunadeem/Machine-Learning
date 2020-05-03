# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 00:25:20 2020

@author: dasth
"""
#Importing Libraries and Data Set:
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

a=pd.read_csv('Wine.csv')

#Checking for Null and Categorical Variable:
b=a.isnull().sum()
'''
c=pd.value_counts(a['Gender'])
'''

#Splitting the Data to Target and Tool:
x=a.iloc[:,0:-1]
y=a.iloc[:,-1]

#Splitting the Data set to Training and Testing Data:
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.80,random_state=0)

#Feature scaling:
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test)

#Dimension Reduction Technique (PCA):
from sklearn.decomposition import PCA
pca=PCA(n_components=None)
x_train=pca.fit_transform(x_train)
x_test=pca.transform(x_test)
variance_list=pca.explained_variance_ratio_
'''
we are considering Maxium 2 Eigen Value, which gives us the 2 Eigen Vector 
having Maximum 2 Variance distribution in those vector
'''
pca=PCA(n_components=2)      # Selecting Top 2 Variance
x_train=pca.fit_transform(x_train)
x_test=pca.transform(x_test)
 
#Fitting the data to Logistic Reg Module:
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(x_train,y_train)

#Prediction of Training Data:
y_predict_train=classifier.predict(x_train)

#Prediction of Testing Data:
y_predict_test=classifier.predict(x_test)

#Confusion Metrix to evaluate the prediction:
from sklearn.metrics import confusion_matrix
met_train=confusion_matrix(y_train, y_predict_train)
met_test=confusion_matrix(y_test, y_predict_test)
Accuracy_train=((met_train[0][0])+(met_train[1][1])+(met_train[2][2]))/(met_train.sum())*100
Accuracy_test=((met_test[0][0])+(met_test[1][1])+(met_test[2][2]))/(met_test.sum())*100

#Plotting the Graph:
#Training Set Graph:
from matplotlib.colors import ListedColormap
clr=ListedColormap(['red','green','skyblue'])
x_set,y_set=x_train,y_train
x1,x2=np.meshgrid(np.arange(x_set[:,0].min()-1,x_set[:,0].max()+1,0.01),
                  np.arange(x_set[:,1].min()-1,x_set[:,1].max()+1,0.01))
y_predict_grid=classifier.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape)
#Plotting Outline :
plt.contourf(x1,x2,y_predict_grid,cmap=clr,alpha=0.7)
#Plotting Scatter Plot:
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set==j,0],x_set[y_set==j,1],c=clr(i),label=j)

plt.title('Training Set and Accuracy of Linear Reg Model with PCA is {:.2f} %'.format(Accuracy_train))
plt.legend()
plt.xlabel('Age')
plt.ylabel('Salary')
plt.show()

#Testing Set Graph:
x_set,y_set=x_test,y_test
x1,x2=np.meshgrid(np.arange(x_set[:,0].min()-1,x_set[:,0].max()+1,0.01),
                  np.arange(x_set[:,1].min()-1,x_set[:,1].max()+1,0.01))
y_predict_grid=classifier.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape)
#Plotting Outline :
plt.contourf(x1,x2,y_predict_grid,cmap=clr,alpha=0.7)
#Plotting Scatter Plot:
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set==j,0],x_set[y_set==j,1],c=clr(i),label=j)

plt.title('Testing Set and Accuracy of Linear Reg Model with PCA is {:.2f} %'.format(Accuracy_test))
plt.legend()
plt.xlabel('Age')
plt.ylabel('Salary')
plt.show()

