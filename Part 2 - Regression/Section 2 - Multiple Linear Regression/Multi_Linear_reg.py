#Importing Libraries and Data Set:
import pandas as pd 

a=pd.read_csv('50_Startups.csv')

#Checking for Null and Categorical Variable:
b=pd.value_counts(a['State'])
c=a.isnull().sum() #check for null value

#Splitting the Data to Target and Tool:
x=a.iloc[:,:4]   # Tool:make sure this is a matrix
y=a.iloc[:,-1]   #Target:make sure this is a Vector

x=pd.get_dummies(x,drop_first=True)

#Encoding Catogorical variable:
x=pd.get_dummies(x,drop_first=True)

#Splitting the Data set to Training and Testing Data:
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=0)

#Fitting the data to Multi Linear Reg Module:
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train, y_train)
#predicting the data
pred_train=reg.predict(x_train)  # used only to plotting graph
pred_test=reg.predict(x_test)

#R2 and Mean Squared Error:
from sklearn.metrics import mean_squared_error,r2_score
R=r2_score(y_test,pred_test)
mse_testing=mean_squared_error(y_test,pred_test)
print('MSE of testing data is {:.2f}'.format(mse_testing))
print('R score is {:.2f}'.format(R))

#To reduce the MSE value, we follow Bacward elimination:
"""
Backward elimination of Independent: to deteermine which independent 
variable is effecting the output in greater scale

Below step is done to satisfy y =b0+b1*x+b2*x+....
in the above equation the valeu of x0=1, hence we add an array of 1 
so that the syatem understands the data is in the for of above equation

Now when we add an array of 1 to X, the column is appended to the last,
hence insted of adding column to x we add x to column, in this way we get
array of 1 in the begining of X 
"""

import statsmodels.api as st
x=st.add_constant(x)
x=x[['const','State_Florida', 'State_New York'
     ,'R&D Spend', 'Administration', 'Marketing Spend']]
x_opt=x.loc[:,['const','State_Florida', 'State_New York'
     ,'R&D Spend', 'Administration', 'Marketing Spend']]
reg_ols=st.OLS(endog=y,exog=x_opt).fit()

"""
#reg_ols.summary() # to be executed in the console
"""

#Eliminating the column with highest P value:Backward Elimination
x_opt=x.loc[:,['const', 'State_New York'
     ,'R&D Spend', 'Administration', 'Marketing Spend']]
reg_ols=st.OLS(endog=y,exog=x_opt).fit()

x_opt=x.loc[:,['const','R&D Spend', 'Administration', 
               'Marketing Spend']]
reg_ols=st.OLS(endog=y,exog=x_opt).fit()

x_opt=x.loc[:,['const','R&D Spend','Marketing Spend']]
reg_ols=st.OLS(endog=y,exog=x_opt).fit()

#Refitting the optimum data to Reg Module to check the MSE value & R2
x_new=x_opt
x_train_n,x_test_n,y_train,y_test=train_test_split(x_new,y,train_size=0.8,random_state=0)

reg=LinearRegression()
reg.fit(x_train_n, y_train)

pred_train=reg.predict(x_train_n)  # used only to plotting graph
pred_test_n=reg.predict(x_test_n)

R_n=r2_score(y_test,pred_test_n)
mse_testing_new=mean_squared_error(y_test,pred_test_n)
print('MSE_new of testing data is {:.2f}'.format(mse_testing_new))
print('R_new score is {:.2f}'.format(R_n))

 
 
 
 
 
 
 
 
 
 
 
 
 
 
 