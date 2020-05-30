# -*- coding: utf-8 -*-
"""
Created on Fri May 29 23:22:14 2020

@author: dasth
"""
'''
On their own, logistic regressions are only binary classifiers, 
meaning they cannot handle target vectors with more than two classes. 
However, there are clever extensions to logistic regression to do just that.
 In one-vs-rest logistic regression (OVR) a separate model is trained for 
 each class predicted whether an observation is that class or 
 not (thus making it a binary classification problem). 
 It assumes that each classification problem (e.g. class 0 or not) 
 is independent.
'''

from sklearn.datasets import load_iris
import numpy as np
from pandas import DataFrame as df
import seaborn as sns

data=load_iris()

a=np.concatenate((data['data'],data['target'].reshape(-1,1)),axis=1)

data['feature_names']

a=df(a,columns=['sepal length (cm)',
 'sepal width (cm)',
 'petal length (cm)',
 'petal width (cm)','Target'])

x=a.iloc[:,:-1]
y=a.iloc[:,-1]

from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x=sc_x.fit_transform(x)

from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0,multi_class='ovr')
classifier.fit(x,y)

sns.pairplot(a,hue='Target')

#Predicting with new set
x_new=[5,4,1,0.5]
predict=classifier.predict(sc_x.transform([x_new]))
output={0:'setosa',1:'versicolor',2:'virginica'}

print('prediction is {}'.format(output[predict[0]]))











