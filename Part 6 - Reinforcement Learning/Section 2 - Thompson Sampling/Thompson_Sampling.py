# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 23:37:18 2020

@author: dasth
"""
#Importing Libraries and Data Set: 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

a=pd.read_csv('Ads_CTR_Optimisation.csv')

#Coding Thompson_Sampling_Algorithm:
d=10                                        # number of adds
Numb_of_reward_1=[0]*d
Numb_of_reward_0=[0]*d
ad_selected=[]
Total_reward=0

#Computing Random Sample:
import random
for n in range (0,len(a)):
    max_Beta=0
    ad=0
    for i in range (0,d):
        random_beta=random.betavariate(Numb_of_reward_1[i]+1,
                                       Numb_of_reward_0[i]+1)
        if random_beta>max_Beta:
            max_Beta=random_beta
            ad=i                
    ad_selected.append(ad)
    reward=a.iloc[n,ad]
    Total_reward+=reward
    if reward==1:
       Numb_of_reward_1[ad]+=1
    else:
        Numb_of_reward_0[ad]+=1

#Plotting the Graph:
plt.hist(ad_selected)
plt.title('Thompson_Sampling of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()
    
