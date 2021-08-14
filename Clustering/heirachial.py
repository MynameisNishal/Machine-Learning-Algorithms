import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset=pd.read_csv('Mall_CUstomers.csv')

X=dataset.iloc[:,[3,4]].values
"""
import scipy.cluster.hierarchy as sch
dendrogram=sch.dendrogram(sch.linkage(X,method='ward'))
"""
from sklearn.cluster import AgglomerativeClustering
hc=AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward')
y_hc=hc.fit_predict(X)


plt.scatter(X[y_hc==0,0], X[y_hc==0,1],s=50,c='red',label='CLuster 1')

plt.scatter(X[y_hc==1,0], X[y_hc==1,1],s=50,c='blue',label='CLuster 2')

plt.scatter(X[y_hc==2,0], X[y_hc==2,1],s=50,c='green',label='CLuster 3')

plt.scatter(X[y_hc==3,0], X[y_hc==3,1],s=50,c='cyan',label='CLuster 4')

plt.scatter(X[y_hc==4,0], X[y_hc==4,1],s=50,c='magenta',label='CLuster 5')
plt.show()