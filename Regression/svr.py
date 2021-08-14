import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset=pd.read_csv('Position_salaries.csv')

X=dataset.iloc[:,1:2].values
Y=dataset.iloc[:,2:3].values

from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
sc_Y=StandardScaler()
X=sc_X.fit_transform(X)
Y=sc_Y.fit_transform(Y)

from sklearn.svm import SVR
regressor=SVR(kernel='rbf')
regressor.fit(X,Y)


y_pred=sc_X.transform(np.array([[6.5]]))
y_pred=regressor.predict(y_pred)
y_pred=sc_Y.inverse_transform(y_pred)

plt.scatter(X,Y,color='red')
plt.plot(X,regressor.predict(X))
