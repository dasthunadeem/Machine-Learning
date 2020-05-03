# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 19:40:41 2020

@author: dasth
"""
#Importing Libraries and Data Set: 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

a=pd.read_csv('Restaurant_Reviews.tsv',quoting=3,delimiter='\t')

reviews=a.iloc[:,0]

#Cleaning the Data Set:
from re import sub
import nltk 
#nltk.download('stopwords')   # if stopwords are not available in ur system
from nltk.corpus import stopwords
stopwords_english=stopwords.words('english')

from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
corpus=[]

for i in range(0,len(reviews)):
    review=sub('[^a-zA-z]',' ',reviews[i])  #Removing the Punctuations
    review=review.lower().split()          #Converting to Lower case and splitting the Words
    '''
    #stoperods.words('english') has list of stopwords in english
    #in case Books "set" will help in increasing the speed of algorithm search
    #ps.stem will creating stemming of each word 
    '''
    review=[ps.stem(word) for word in review if not word in set(stopwords_english)]
    
    review=' '.join(review) # Joining the data set back to String
    corpus.append(review)

#Creating Bag of Words Model:
'''
Bag of Words Model Consist all the Unique words and converting each word
to a coloumn, 
each review will be having many columns of Unique word 
In short:
it will be the input for Sparse matrix 
'''
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500)
x=cv.fit_transform(corpus).toarray()    # Independent Variable

#Depedent Variable:
y=a.iloc[:,1]

#Fitting the data to Classification Module:
'''
Naive Bayes Classifier:
'''
#Splitting the Data set to Training and Testing Data:
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=0)

#Fitting the data to Naive Bayes Module:
from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(x_train,y_train)

#Prediction of Training Data:
y_predict_train=classifier.predict(x_train)

#Prediction of Testing Data:
y_predict_test=classifier.predict(x_test)

#Confusion Metrix to evaluate the prediction:
from sklearn.metrics import confusion_matrix
met_train=confusion_matrix(y_train, y_predict_train)
met_test=confusion_matrix(y_test, y_predict_test)
    
accuracy=((met_test[0][0])+(met_test[:,1][1]))/(met_test.sum())*100

print('Accuracy of this model {:.2f} %'.format(accuracy))

    
    
    
    
    
    
    
    
    
    
    
