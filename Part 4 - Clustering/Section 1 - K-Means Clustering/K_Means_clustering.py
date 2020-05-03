# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 15:26:03 2020

@author: dasth
"""
#Importing Libraries and Data Set:
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

a=pd.read_csv('Mall_Customers.csv')

#Checking for Null and Categorical Variable:
b=a.isnull().sum()

x=a.iloc[:,3:]

#Find Optimal number of Clusters (K) using Elbow Method:
from sklearn.cluster import KMeans
WCSS=[]
for i in range (1,11):       #since we are consideriing 10 clusters 
    km=KMeans(n_clusters=i,init='k-means++',n_init=10,max_iter=300,random_state=0)
    km.fit(x)
    WCSS.append(km.inertia_)  
#Plotting Elbow Graph:
plt.plot(range(1,11),WCSS)
plt.title('Elbow Plot for Optimum Cluster (K)')
plt.xlabel('Number of Cluster (K)'   )
plt.ylabel('WCSS')
plt.show()
    
#Fitting and predicting the data to K-Mean Cluster Module:
km=KMeans(n_clusters=5,init='k-means++',n_init=10,max_iter=300,random_state=0)
y_km=km.fit_predict(x)

#Plotting The cluster:
plt.scatter(x.iloc[y_km==0,0],x.iloc[y_km==0,1],s=100,alpha=0.8,c='red',label='Carefull_spend')
plt.scatter(x.iloc[y_km==1,0],x.iloc[y_km==1,1],s=100,alpha=0.8,c='green',label='Standard_cust')
plt.scatter(x.iloc[y_km==2,0],x.iloc[y_km==2,1],s=100,alpha=0.8,c='blue',label='Target_cust')
plt.scatter(x.iloc[y_km==3,0],x.iloc[y_km==3,1],s=100,alpha=0.8,c='navy',label='careless_spend')
plt.scatter(x.iloc[y_km==4,0],x.iloc[y_km==4,1],s=100,alpha=0.8,c='violet',label='sensible_cust')

plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],s=300,c='gold')

plt.title('K_means Cluster Plot')
plt.xlabel('Salary')
plt.ylabel('Customer Rating')
plt.legend()
plt.show()
