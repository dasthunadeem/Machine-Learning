# -*- coding: utf-8 -*-
"""
Created on Sat May 30 18:37:07 2020

@author: dasth
"""

#Importing Libraries and Data Set:
import pandas as pd
import seaborn as sns
import numpy as np
from pandas import DataFrame as df

a=pd.read_csv('Purchased_Dataset.csv',
              usecols=['Age', 'EstimatedSalary', 'Purchased'])

#Checking for Null and Categorical Variable:
sns.heatmap(a.isnull(),yticklabels=False,cbar=False)

#Splitting the Data to Target and Tool:
x=a.iloc[:,0:-1]
y=a.iloc[:,-1]

#Splitting the Data set to Training and Testing Data:
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,
                                               y,
                                               train_size=0.75,
                                               random_state=0)

#Feature Scaling the data :
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test)

#Fitting the model to Classification Algorithm:
from sklearn.linear_model import LogisticRegression
model_logistic=LogisticRegression()


from sklearn.neighbors import KNeighborsClassifier
model_knn=KNeighborsClassifier(n_neighbors=5)


from sklearn.naive_bayes import GaussianNB
model_bayes=GaussianNB()


#List of Classifiers:
list_algo=[model_logistic,model_knn,model_bayes]

#K cross Validation for selection of Best Model:
def Best_Model(list_classification,scoring):
    '''  

    Parameters
    ----------
    d : TYPE
        List of Classification algorithm.
        
    SCOREING:
        SCORE SELECTION ACCURACY OR ....

    Returns
    -------
    data : TYPE
        return with the accuracy score.

    '''
    from sklearn.model_selection import cross_val_score
    accu_score=[]
    model_select=[]
    
    for x in list_classification:
        x.fit(x_train,y_train)
        k_cross=cross_val_score(estimator=x,
                        X=x_train, y=y_train,
                        scoring=scoring, cv=10, n_jobs=-1)
        model_select.append((x))
        accu_score.append((k_cross.mean())*100)
    
    data=np.array([model_select,accu_score]).T
    data=df(data,columns=['Model','Accuracy'])
    return data
        
result=Best_Model(list_algo,'accuracy')
















