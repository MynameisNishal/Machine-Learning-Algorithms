# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 02:39:50 2020

@author: Nishal Sundarraman
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset=pd.read_csv('train.csv')

temp=dataset.iloc[:,:].values

from sklearn.impute import SimpleImputer
si=SimpleImputer(missing_values=np.nan,strategy='mean')
si.fit(temp[:,[5]])
temp[:,[5]]=si.transform(temp[:,[5]])

si1=SimpleImputer(missing_values=np.nan,strategy='most_frequent')
si1.fit(temp[:,[11]])
temp[:,[11]]=si1.transform(temp[:,[11]])

temp=pd.DataFrame(temp)

Y=dataset.iloc[:,[1]].values
X=temp.iloc[:,[2,4,5,6,7,11]].values

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
X[:,1]=le.fit_transform(X[:,1])
le1=LabelEncoder()
X[:,5]=le1.fit_transform(X[:,5])


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

from sklearn.decomposition import PCA
pca=PCA(n_components=None)
X_train=pca.fit_transform(X_train)
X_test=pca.transform(X_test)
explained_variance=pca.explained_variance_ratio_
"""
from sklearn.svm import SVC
classifier=SVC()
classifier.fit(X_train,Y_train)
"""
"""
from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=100,metric='minkowski',p=2)
classifier.fit(X_train,Y_train)
"""

from xgboost import XGBClassifier

from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(X_train,Y_train)

y_pred=classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test,y_pred)

from sklearn.metrics import accuracy_score
score=accuracy_score(Y_test,y_pred)