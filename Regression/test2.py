import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset=pd.read_csv('Position_Salaries.csv')

X=dataset.iloc[:,1:2].values
Y=dataset.iloc[:,2].values

from sklearn.preprocessing import PolynomialFeatures
pl_reg=PolynomialFeatures(degree=4)
X_poly=pl_reg.fit_transform(X)

from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X_poly,Y)

plt.scatter(X, Y,color='red')
plt.plot(X,lin_reg.predict(X_poly))
