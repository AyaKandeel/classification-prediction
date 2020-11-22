import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
df=pd.read_excel('online_retail.xlsx',sheet_name='Online Retail')
df.head()

# Data clean up
df=df.loc[df['Quantity']>0] 
df =df[pd.notnull(df['CustomerID'])] 
df=df.loc[df['InvoiceDate']<'2011-12-01']

 # total of sales
df['Sales']=df['Quantity']*df['UnitPrice']
df.head()
customer_df = df.groupby('CustomerID').agg({'Sales': sum,'InvoiceNo': lambda x: x.nunique()})

# Select the columns we want to use
customer_df.columns = ['TotalSales', 'OrderCount']

# create a new column 'AvgOrderValu'
customer_df['AvgOrderValue'] = customer_df['TotalSales'] / customer_df['OrderCount']
customer_df.head()
rank_df=customer_df.rank(method='first')
normalized_df=(rank_df-rank_df.mean())/rank_df.std()
normalized_df.head(10)
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import euclidean_distances
for n_cluster in [4,5,6,7,8]:
   kmeans=KMeans(n_clusters=n_cluster).fit(normalized_df[['TotalSales', 'OrderCount', 'AvgOrderValue']])
silhouette_avg=silhouette_score(normalized_df[['TotalSales', 'OrderCount', 'AvgOrderValue']],kmeans.labels_)
print('Silhouette score for %i Clusters :%0.4f'%(n_cluster,silhouette_avg))
from sklearn import cluster
import numpy as np
sse=[]
krange = list(range(2,11))
X= normalized_df[['TotalSales', 'OrderCount', 'AvgOrderValue']].values
for n in krange:
   model=cluster.KMeans(n_clusters=n,random_state=3)
   model.fit_predict(X)
   cluster_assignments=model.labels
   centers=model.cluster_centers
   sse.append(np.sum((X-centers[cluster_assignments])**2))
plt.plot(krange,sse)
plt.xlabel('$K$')
plt.ylabel('sum of Squares')
plt.show()
kmeans=KMeans(n_clusters=4).fit(normalized_df[['TotalSales', 'OrderCount', 'AvgOrderValue']])
four_cluster_df=normalized_df[['TotalSales', 'OrderCount', 'AvgOrderValue']].copy(deep=True)
four_cluster_df['Cluster']=kmeans.labels
four_cluster_df.head(10)
cluster0_metrics=kmeans.cluster_centers_[0]
cluster1_metrics=kmeans.cluster_centers_[1]
cluster2_metrics=kmeans.cluster_centers_[2]
cluster3_metrics=kmeans.cluster_centers_[3]
data=[cluster0_metrics,cluster1_metrics,cluster2_metrics,cluster3_metrics]
cluster_center_df=pd.DataFrame(data)
cluster_center_df.colums=four_cluster_df.columns[0:3]
four_cluster_df
plt.scatter(
four_cluster_df.loc[four_cluster_df['Cluster']==0]['OrderCount'],four_cluster_df.loc[four_cluster_df['Cluster']==0]['TotalSales'],c='b')
plt.scatter(four_cluster_df.loc[four_cluster_df['Cluster']==1]['OrderCount'],four_cluster_df.loc[four_cluster_df['Cluster']==1]['TotalSales'],c='r')
plt.scatter(four_cluster_df.loc[four_cluster_df['Cluster']==2]['OrderCount'],four_cluster_df.loc[four_cluster_df['Cluster']==2]['TotalSales'],c='orange')
plt.scatter(four_cluster_df.loc[four_cluster_df['Cluster']==3]['OrderCount'],four_cluster_df.loc[four_cluster_df['Cluster']==3]['TotalSales'],c='g')
plt.title('AvgOrderValue.TotalSales Cluster')
plt.xlabel('TotalSales')
plt.ylabel('AvgOrderValue')
plt.grid()
plt.show()
plt.scatter(four_cluster_df.loc[four_cluster_df['Cluster']==0]['OrderCount'],four_cluster_df.loc[four_cluster_df['Cluster']==0]['TotalSales'],c='b')
plt.scatter(four_cluster_df.loc[four_cluster_df['Cluster']==1]['OrderCount'],four_cluster_df.loc[four_cluster_df['Cluster']==1]['TotalSales'],c='r')
plt.scatter(four_cluster_df.loc[four_cluster_df['Cluster']==2]['OrderCount'],four_cluster_df.loc[four_cluster_df['Cluster']==2]['TotalSales'],c='orange')
plt.scatter(four_cluster_df.loc[four_cluster_df['Cluster']==3]['OrderCount'],four_cluster_df.loc[four_cluster_df['Cluster']==3]['TotalSales'],c='g')
plt.title('AvgOrderValue.TotalSales Cluster')
plt.xlabel('TotalSales')
plt.ylabel('AvgOrderValue')
plt.grid()
plt.show()
plt.scatter(four_cluster_df.loc[four_cluster_df['Cluster']==0]['OrderCount'],four_cluster_df.loc[four_cluster_df['Cluster']==0]['AvgOrderValue'],c='b')
plt.scatter(four_cluster_df.loc[four_cluster_df['Cluster']==1]['OrderCount'],four_cluster_df.loc[four_cluster_df['Cluster']==1]['AvgOrderValue'],c='r')
plt.scatter(four_cluster_df.loc[four_cluster_df['Cluster']==2]['OrderCount'],four_cluster_df.loc[four_cluster_df['Cluster']==2]['AvgOrderValue'],c='orange')
plt.scatter(four_cluster_df.loc[four_cluster_df['Cluster']==3]['OrderCount'],four_cluster_df.loc[four_cluster_df['Cluster']==3]['AvgOrderValue'],c='g')
plt.title('AvgOrderValue.TotalSales Cluster')
plt.xlabel('TotalSales')
plt.ylabel('AvgOrderValue')
plt.grid()
plt.show()
plt.scatter(four_cluster_df.loc[four_cluster_df['Cluster']==0]['TotalSales'],four_cluster_df.loc[four_cluster_df['Cluster']==0]['AvgOrderValue'],c='b')
plt.scatter(four_cluster_df.loc[four_cluster_df['Cluster']==1]['TotalSales'],four_cluster_df.loc[four_cluster_df['Cluster']==1]['AvgOrderValue'],c='r')
plt.scatter(four_cluster_df.loc[four_cluster_df['Cluster']==2]['TotalSales'],four_cluster_df.loc[four_cluster_df['Cluster']==2]['AvgOrderValue'],c='orange')
plt.scatter(four_cluster_df.loc[four_cluster_df['Cluster']==3]['TotalSales'],four_cluster_df.loc[four_cluster_df['Cluster']==3]['AvgOrderValue'],c='g')
plt.title('AvgOrderValue.TotalSales Cluster')
plt.xlabel('TotalSales')
plt.ylabel('AvgOrderValue')
plt.grid()
plt.show()
high_value_cluster=four_cluster_df.loc[four_cluster_df['Cluster']==2]
pd.DataFrame(df.loc[df['CustomerID'].isin(high_value_cluster.index)].groupby('Description').count()['StockCode'].sort_values(ascending=False).head())
    
    
    
    
    
    
    
    
    