# -*- coding: utf-8 -*-
#Importing Libraries and Data Set: 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

a=pd.read_csv('Ads_CTR_Optimisation.csv')

#Coding UCB_Algorithm:
d=10                                        # number of adds
Numb_of_selection=[0]*d
Sum_of_rewards=[0]*d
ad_selected=[]
Total_reward=0

#Computing UCB(Upper Confidence Bound):
from math import sqrt,log
for n in range (0,len(a)):
    max_UCB=0
    ad=0
    for i in range (0,d):
        if Numb_of_selection[i]>0:       
            avg_reward=Sum_of_rewards[i]/Numb_of_selection[i] 
            delta_i=sqrt(1.5*(log(n+1)/Numb_of_selection[i])) 
            UCB=avg_reward+delta_i
        else:
            UCB=1e400        
        if UCB>max_UCB:
            max_UCB=UCB
            ad=i
    ad_selected.append(ad)
    Numb_of_selection[ad]+=1
    reward=a.iloc[n,ad]
    Sum_of_rewards[ad]+=reward
    Total_reward+=reward

#Plotting the Graph:
plt.hist(ad_selected)
plt.title('UCB of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()

'''
Note:
P.S:
if the add is selected and has a reward then at the end of current round,
in the next round 'avg_reward' will increase the UCB
(since UCB=avg_reward+delta) and thus making it MAX UCB 

if this doesnot have reward then in next round 'delta' will give the opprtnty 
to other add's which has least selection,
(i,e delta=log()/4 compared log()/3 add with 3 will get a chance)
thus providing apportunity to all add untill the point wer delta
will remain same for all the add and add with previous reward will go forward

with large number of sample, the 'final output' will tend to 
'Expected Value' as per 'Law of Large Number'

in short:
delta will increase only if the particular add selected is less
compared to others
avg_reward will increase if it had a reward in its previous round 

'''
