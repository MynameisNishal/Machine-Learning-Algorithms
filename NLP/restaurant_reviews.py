import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset=pd.read_csv('Restaurant_Reviews.tsv',delimiter='\t',quoting=3)

import re
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
corpus=[]

for i in range(0,1000):
        
    review=re.sub('[^-a-zA-Z]',' ' ,dataset['Review'][i])
    review=review.lower()
    review=review.split()
    review=[ps.stem(word) for word in review if not word in set(stopwords.words('english')) ]
    review=' '.join(review)
    corpus.append(review)
    
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500)
X=cv.fit_transform(corpus).toarray()

Y=dataset.iloc[:,1].values



from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y , test_size=0.1,random_state=0)

"""
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
"""
"""
from sklearn.decomposition import PCA
pca=PCA(n_components=83)
X_train=pca.fit_transform(X_train)
X_test=pca.transform(X_test)
explained_variance=pca.explained_variance_ratio_"""

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, Y_train)

test='Restaurant is lacking good food services good food'
Y_pred=classifier.predict(test)

#from sklearn.metrics import confusion_matrix
#cm=confusion_matrix(Y_test, Y_pred)

#from sklearn.metrics import accuracy_score
#score=accuracy_score(Y_test,Y_pred)