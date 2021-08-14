import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np

dataset=pd.read_csv('Position_Salaries.csv')

X=dataset.iloc[:,1:2].values
Y=dataset.iloc[:,2].values

from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X,Y)

from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=3)
X_poly=poly_reg.fit_transform(X)

lin_reg_2=LinearRegression()
lin_reg_2.fit(X_poly,Y)
"""
plt.scatter(X,Y,color='red')
plt.plot(X,lin_reg.predict(X), color='blue')
plt.title('Truth vs Lie(Linear Regression')
plt.xlabel('Level')
plt.ylabel('Salary')"""

plt.scatter(X,Y,color='red')
plt.plot(X,lin_reg_2.predict(poly_reg.fit_transform(X)))
plt.title('Truth vs Lie(Polynomial Regression')
plt.xlabel('Level')
plt.ylabel('Salary')

