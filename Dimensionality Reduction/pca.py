import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset=pd.read_csv('wine.csv')
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1:].values

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

from sklearn.decomposition import PCA
pca=PCA(n_components=2)
X_train=pca.fit_transform(X_train)
X_test=pca.transform(X_test)
variance=pca.explained_variance_ratio_



from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)  
classifier.fit(X_train,Y_train)

y_pred=classifier.predict(X_test)

from sklearn.metrics import accuracy_score
score=accuracy_score(Y_test, y_pred)