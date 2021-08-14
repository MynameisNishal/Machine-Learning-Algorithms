import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset=pd.read_csv('Position_Salaries.csv')

X=dataset.iloc[:,1:2].values
Y=dataset.iloc[:,2].values

from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=10000,random_state=0)
regressor.fit(X, Y)

X_grid=np.arange(min(X),max(X),0.01)
X_grid=X_grid.reshape(len(X_grid),1)

y_pred=regressor.predict([[6.5]])

plt.scatter(X,Y,color='red')
plt.plot(X_grid,regressor.predict(X_grid),color='blue')