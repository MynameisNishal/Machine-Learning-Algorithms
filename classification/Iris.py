import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset=pd.read_csv('Data.csv')
X=dataset.iloc[:,:-1]
Y=dataset.iloc[:,3]



x=X.iloc[:,1:3].values
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan,strategy="mean",)
imputer.fit(x)
x=imputer.transform(x)
X=pd.DataFrame(X)
X=X.drop(columns=['Age','Salary'])
x=pd.DataFrame(x)
x=x.rename(columns={0:"Age",1:"Salary"} )
X=pd.concat([X,x],axis=1)


a=dataset.iloc[:,0]
from sklearn import preprocessing 
le_x=preprocessing.LabelEncoder()
a=le_x.fit_transform(a)
a=pd.DataFrame(a)
a=a.rename(columns={0:"Country"})
X=X.drop(columns=['Country'])
X=pd.concat([a,X],axis=1)

le_y=preprocessing.LabelEncoder()
Y=le_y.fit_transform(Y)

oe=preprocessing.OneHotEncoder(handle_unknown='ignore')

a=oe.fit_transform(a).toarray()
a=pd.DataFrame(a)
X=X.drop(columns='Country')
X=pd.concat([a,X],axis=1)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.2 , random_state=0)

from sklearn import preprocessing
sc_X= preprocessing.StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)











