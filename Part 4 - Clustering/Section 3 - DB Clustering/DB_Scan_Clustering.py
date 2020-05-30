# -*- coding: utf-8 -*-
"""
Created on Sat May 30 01:02:55 2020

@author: dasth
"""

import pandas as pd 
import matplotlib.pyplot as plt

a=pd.read_csv('Mall_Customers.csv')
x=a.iloc[:,3:]

from sklearn.cluster import DBSCAN
clust=DBSCAN(eps=3, min_samples=4)
clust=clust.fit(x) 

y_clust=clust.fit_predict(x)
'''
to find the bumber of Cluster, ignore -1 , since it is the noise
'''
b=pd.value_counts(y_clust)

plt.scatter(x.iloc[y_clust==0,0],x.iloc[y_clust==0,1],s=50,alpha=0.8,
            c='red')
plt.scatter(x.iloc[y_clust==1,0],x.iloc[y_clust==1,1],s=50,alpha=0.8,
            c='green')
plt.scatter(x.iloc[y_clust==2,0],x.iloc[y_clust==2,1],s=50,alpha=0.8,
            c='blue')
plt.scatter(x.iloc[y_clust==3,0],x.iloc[y_clust==3,1],s=50,alpha=0.8,
            c='navy')
plt.scatter(x.iloc[y_clust==4,0],x.iloc[y_clust==4,1],s=50,alpha=0.8,
            c='#7f7f7f')
plt.scatter(x.iloc[y_clust==5,0],x.iloc[y_clust==5,1],s=50,alpha=0.8,
            c='#8c564b')
plt.scatter(x.iloc[y_clust==6,0],x.iloc[y_clust==6,1],s=50,alpha=0.8,
            c='#9467bd')
plt.scatter(x.iloc[y_clust==7,0],x.iloc[y_clust==7,1],s=50,alpha=0.8,
            c='#d62728')
plt.scatter(x.iloc[y_clust==8,0],x.iloc[y_clust==8,1],s=50,alpha=0.8,
            c='#17becf')
plt.scatter(x.iloc[y_clust==-1,0],x.iloc[y_clust==-1,1],s=50,alpha=0.8,
            c='black')
plt.title('K_means Cluster Plot\n blasck are ouliers')
plt.xlabel('Salary')
plt.ylabel('Customer Rating')
plt.legend()
plt.show()
