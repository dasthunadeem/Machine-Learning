# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 00:38:16 2020

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

#Find Optimal number of Clusters (K) using Dendogram Method:
import scipy.cluster.hierarchy as sch
link=sch.linkage(x,method='ward')   # Creating Link between all data points
dend=sch.dendrogram(link)  # Plotting Dendogram
plt.title('Dendogram Chart for Optimum Cluster (K)')
plt.xlabel('Customer'   )
plt.ylabel('Eucledian Distance')
plt.show()
    
#Fitting and predicting the data to Hierarchical Cluster Module:
from sklearn.cluster import AgglomerativeClustering
hc=AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage="ward")
y_hc=hc.fit_predict(x)

#Plotting The cluster:
plt.scatter(x.iloc[y_hc==0,0],x.iloc[y_hc==0,1],alpha=0.9,c='red',label='Carefull_spend')
plt.scatter(x.iloc[y_hc==1,0],x.iloc[y_hc==1,1],alpha=0.9,c='green',label='Standard_cust')
plt.scatter(x.iloc[y_hc==2,0],x.iloc[y_hc==2,1],alpha=0.9,c='blue',label='Target_cust')
plt.scatter(x.iloc[y_hc==3,0],x.iloc[y_hc==3,1],alpha=0.9,c='navy',label='Careless_spend')
plt.scatter(x.iloc[y_hc==4,0],x.iloc[y_hc==4,1],alpha=0.9,c='violet',label='sensible_cust')

plt.title('Hierarchical Cluster Plot')
plt.xlabel('Salary')
plt.ylabel('Customer Rating')
plt.legend()
plt.show()


