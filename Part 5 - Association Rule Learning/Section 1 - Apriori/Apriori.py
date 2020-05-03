# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 01:09:15 2020

@author: dasth
"""
#Importing Libraries and Data Set:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

a=pd.read_csv('Market_Basket_Optimisation.csv',header=None)

#Converting Data Set to List:
transact=[]
for x in range(0,len(a)):
    transact.append([str(a.iloc[x,y]) for y in range(0,len(a.iloc[x]))])

#Fitting the Data to Apriori Module:
from apyori import apriori
rules=apriori(transact,min_support=0.003,min_confidence=0.2,min_lift=3.0,min_length=2)

#Visulising the Result:
result=list(rules)
output =[]

for i in range (0,len(result)):
    output.append(['Rule:\t'+str(result[i][2][0][0]),
                    'Effect:\t'+str(result[i][2][0][1]),
                    'Support:\t'+str(result[i][1]),
                    'Confidence:\t'+str(result[i][2][0][2]),
                    'Lift:\t'+str(result[i][2][0][3])])


        
