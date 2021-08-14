import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset=pd.read_csv('iris.csv')

X=dataset.iloc[:,[1,2,3,4]].values
Y=dataset.iloc[:,[5]].values

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
Y=le.fit_transform(Y)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.13 , random_state=42)

from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(X_train,Y_train)

Y_pred=classifier.predict(X_test)

from sklearn.metrics import accuracy_score
score=accuracy_score(Y_test, Y_pred)

import seaborn as sns

#sns.barplot(x='type',y='x2',data=dataset)
sns.scatterplot(x='type',y=Y_pred,data=Y_pred)