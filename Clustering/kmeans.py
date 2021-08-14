import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset=pd.read_csv('Mall_Customers.csv')

X=dataset.iloc[:,[3,4]].values

wcss=[]
from sklearn.cluster import KMeans
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',n_init=10,max_iter=300,random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    

kmeans=KMeans(n_clusters=5,init='k-means++',n_init=10,max_iter=300,random_state=0)
y_pred=kmeans.fit_predict(X)

plt.scatter(X[y_pred==0,0], X[y_pred==0,1],s=50,c='red',label='CLuster 1')

plt.scatter(X[y_pred==1,0], X[y_pred==1,1],s=50,c='blue',label='CLuster 2')

plt.scatter(X[y_pred==2,0], X[y_pred==2,1],s=50,c='green',label='CLuster 3')

plt.scatter(X[y_pred==3,0], X[y_pred==3,1],s=50,c='cyan',label='CLuster 4')

plt.scatter(X[y_pred==4,0], X[y_pred==4,1],s=50,c='magenta',label='CLuster 5')

#plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster.centers_[:,1],s=300,c='yellow',label='Centroids')
