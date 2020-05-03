#Importing Libraries and Data Set:
import numpy as np
import pandas as pd

a=pd.read_csv('Data_Process.csv')  #importing data set:

#Checking for Null and Categorical Variable:
b=pd.value_counts(a['Country'])  #Categorical Variable
c=a.isnull().sum() #Null Value

#Splitting the Data to Target and Tool:
x=a.iloc[:,:3]  # Tool:make sure this is a matrix
y=a.iloc[:,-1]  #Target:make sure this is a Vector

#Nullifying the Missing Value:
from sklearn.impute import SimpleImputer
impute_x=SimpleImputer(missing_values=np.nan,strategy='mean')
x.iloc[:,1:]=impute_x.fit_transform(x.iloc[:,1:])

#Encoding Catogorical variable :
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
#Single Encoder
encode_y=LabelEncoder()
y=encode_y.fit_transform(y)
#Dummy Encoder
encode_x=ColumnTransformer([('0',OneHotEncoder(),[0])],remainder='passthrough')
x=encode_x.fit_transform(x)

#Splitting the Data set to Training and Testing Data:
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,
                                               random_state=0)

y_train=y_train.reshape(-1, 1) # Reshaping for 2D array to fit to Standerd Scale

#Feature scaling:
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
sc_y=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test)
y_train=sc_y.fit_transform(y_train)

y_train=y_train.ravel() # Reshapping for 1D array to fit to Reg.Module





















