import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset=pd.read_csv('50_Startups.csv')
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,4].values

from sklearn.preprocessing import LabelEncoder , OneHotEncoder
from sklearn.compose import ColumnTransformer
le=LabelEncoder()
X[:,3]=le.fit_transform(X[:,3])

ct=ColumnTransformer([('one_hot_encoder', OneHotEncoder(categories='auto'),[3])],remainder='passthrough')
X=ct.fit_transform(X)                      


X=X[:,1:]

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train, Y_train)

import statsmodels.api as sm
X=np.append(np.ones((50,1),dtype=int),X,axis=1)
X_opt=X[:,[0,1,2,3,4,5]]
X_opt=np.array(X_opt,dtype=float)
regressor_OLS=sm.OLS(endog=Y,exog=X_opt).fit()
regressor_OLS.summary()

X_opt=X[:,[0,1,3,4,5]]
X_opt=np.array(X_opt,dtype=float)
regressor_OLS=sm.OLS(endog=Y,exog=X_opt).fit()


X_opt=X[:,[0,3,4,5]]
X_opt=np.array(X_opt,dtype=float)
regressor_OLS=sm.OLS(endog=Y,exog=X_opt).fit()


X_opt=X[:,[0,3,5]]
X_opt=np.array(X_opt,dtype=float)
regressor_OLS=sm.OLS(endog=Y,exog=X_opt).fit()


X_opt=X[:,[0,3,5]]
X_opt=np.array(X_opt,dtype=float)
regressor_OLS=sm.OLS(endog=Y,exog=X_opt).fit()

X_opt=X[:,[0,3]]
X_opt=np.array(X_opt,dtype=float)
regressor_OLS=sm.OLS(endog=Y,exog=X_opt).fit()

























