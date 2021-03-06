import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset=pd.read_csv('Social_Network_Ads.csv')

X=dataset.iloc[:,[2,3]].values
Y=dataset.iloc[:,4].values

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)
classifier.fit(X_train,Y_train)

y_pred=classifier.predict(X_test)

from sklearn.metrics import accuracy_score
score=accuracy_score(Y_test, y_pred)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test, y_pred)


