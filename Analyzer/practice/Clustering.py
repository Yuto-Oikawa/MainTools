import pandas as pd
import numpy as np
from sklearn import cluster

dir = 'data_clustering/'
data1 = pd.read_csv(dir+'気温.csv',index_col=0)
data2 = pd.read_csv(dir+'DB.csv',index_col=0)
data3 = pd.read_csv(dir+'三国志.csv',index_col=0)
data = data2

# 最短距離法
model1 = cluster.AgglomerativeClustering(n_clusters=3,linkage='single')
# 最長距離法
model2 = cluster.AgglomerativeClustering(n_clusters=2,linkage='complete')
# 群平均法
model3 = cluster.AgglomerativeClustering(n_clusters=3,linkage='average')
# k-means
model4 = cluster.KMeans(n_clusters=2)

model = model4
model.fit(data)
print(model.labels_)