import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset=pd.read_csv('Social_Network_Ads.csv')

X=dataset.iloc[:,[2,3]].values
Y=dataset.iloc[:,[4]].values

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y, test_size=0.25,random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=6,metric='minkowski',p=2)
classifier.fit(X_train, Y_train)

Y_pred=classifier.predict(X_test)

from sklearn.metrics import accuracy_score , confusion_matrix
score=accuracy_score(Y_test, Y_pred)

cm=confusion_matrix(Y_test, Y_pred)
