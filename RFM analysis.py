#!/usr/bin/env python
# coding: utf-8

# <a id="1"></a> <br>
# ## Step 1 : Data Understanding

# In[9]:


# Importar as libs necessárias

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt

# import required libraries for clustering
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import cut_tree

from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering


# In[10]:


retail = pd.read_csv('ecommerce.csv', sep=",",encoding="ISO-8859-1", header=0)
retail = retail[['InvoiceNo','Description','InvoiceDate','Quantity','UnitPrice','CustomerID']]
retail = retail.rename(columns = {'InvoiceNo': "id_pedido",
                                    'Description': "descricao",
                                    'InvoiceDate': "data_pgto",
                                    'Quantity' : "quantidade",
                                    'UnitPrice' : "preco_unitario",
                                    'CustomerID': "id_cliente"})
retail.head()


# In[11]:


# df info

retail.info()


# In[12]:


# df descrição

retail.describe()


# <a id="2"></a> <br>
# ## Step 2 : Data Cleaning

# In[13]:


# Calcular % de nulos

df_null = round(100*(retail.isnull().sum())/len(retail), 2)
df_null


# In[14]:


# Droping rows having missing values

retail = retail.dropna()
retail.shape


# In[15]:


retail.describe()


# In[16]:


# Transformar id_cliente em string

retail['id_cliente'] = retail['id_cliente'].astype(str)


# <a id="3"></a> <br>
# ## Step 3 : Data Preparation

# #### We are going to analysis the Customers based on below 3 factors:
# - R (Recency): Number of days since last purchase
# - F (Frequency): Number of tracsactions
# - M (Monetary): Total amount of transactions (revenue contributed)

# In[17]:


retail.head()


# In[18]:


# Novo atributo : Ticket Médio
rfm_m = retail.groupby(['id_cliente']).agg({"preco_unitario": np.sum})
rfm_m.dtypes


# In[19]:


rfm_m.head()


# In[20]:


rfm_m = retail.groupby(['id_cliente']).agg({"preco_unitario": np.sum,"id_pedido": lambda x : x.nunique()})
rfm_m['id_pedido'] = rfm_m['id_pedido'].astype(float)
rfm_m.head()


# In[21]:


# Novo atributo : Ticket Médio
rfm_m['ticket_medio'] = rfm_m['preco_unitario']/rfm_m['id_pedido']
rfm_m.head()


# In[22]:


# Novo atributo : Frequência

rfm_f = retail.groupby('id_cliente')['id_pedido'].count()
rfm_f = rfm_f.reset_index()
rfm_f.columns = ['id_cliente', 'frequencia']
rfm_f.head()


# In[23]:


# Unindo os 2 datasets

rfm = pd.merge(rfm_m, rfm_f, on='id_cliente', how='inner')
rfm.head()


# In[24]:


# Novo atributo : Recência

# Convert to datetime to proper datatype

retail['data_pgto'] = pd.to_datetime(retail['data_pgto'])


# In[25]:


# Compute the maximum date to know the last transaction date

today = pd.Timestamp.today()
today


# In[26]:


retail = retail[retail.data_pgto<=today]


# In[27]:


retail.head()


# In[28]:


# Compute the difference between max date and transaction date

retail['Diff'] = today - retail['data_pgto']
retail.head()


# In[29]:


# Compute last transaction date to get the recency of customers

rfm_p = retail.groupby('id_cliente')['Diff'].min()
rfm_p = rfm_p.reset_index()
rfm_p.head()


# In[30]:


# Extract number of days only

rfm_p['Diff'] = rfm_p['Diff'].dt.days
rfm_p.head()


# In[31]:


rfm = rfm[['id_cliente','ticket_medio','frequencia']]


# In[32]:


# Merge tha dataframes to get the final RFM dataframe

rfm = pd.merge(rfm, rfm_p, on='id_cliente', how='inner')
rfm.columns = ['id_cliente', 'ticket_medio', 'frequencia', 'recencia']
rfm.head()


# #### There are 2 types of outliers and we will treat outliers as it can skew our dataset
# - Statistical
# - Domain specific

# In[33]:


# Outlier Analysis of Amount Frequency and Recency

attributes = ['ticket_medio','frequencia','recencia']
plt.rcParams['figure.figsize'] = [10,8]
sns.boxplot(data = rfm[attributes], orient="v", palette="Set2" ,whis=1.5,saturation=1, width=0.7)
plt.title("Outliers Variable Distribution", fontsize = 14, fontweight = 'bold')
plt.ylabel("Range", fontweight = 'bold')
plt.xlabel("Attributes", fontweight = 'bold')


# In[34]:


#Retirando 5% de outlier
rfm=rfm[rfm['ticket_medio'] < rfm['ticket_medio'].quantile(.95)]
rfm=rfm[rfm['frequencia'] < rfm['frequencia'].quantile(.95)]
rfm=rfm[rfm['recencia'] < rfm['recencia'].quantile(.95)]


# In[35]:


attributes = ['ticket_medio','frequencia','recencia']
plt.rcParams['figure.figsize'] = [10,8]
sns.boxplot(data = rfm[attributes], orient="v", palette="Set2" ,whis=1.5,saturation=1, width=0.7)
plt.title("Outliers Variable Distribution", fontsize = 14, fontweight = 'bold')
plt.ylabel("Range", fontweight = 'bold')
plt.xlabel("Attributes", fontweight = 'bold')


# In[36]:


rfm[['id_cliente','ticket_medio','frequencia','recencia']]


# ### Rescaling the Attributes
# 
# It is extremely important to rescale the variables so that they have a comparable scale.|
# There are two common ways of rescaling:
# 
# 1. Min-Max scaling
# 2. Standardisation (mean-0, sigma-1)
# 
# Here, we will use Standardisation Scaling.

# In[37]:


from sklearn.preprocessing import MinMaxScaler

rfm_df = rfm[['ticket_medio', 'frequencia','recencia']]

scaler = MinMaxScaler()
rfm_df_scaled = scaler.fit_transform(rfm_df)
# dimensiona e traduz cada característica individualmente de tal forma que
#  esteja no intervalo dado no conjunto de treinamento, por exemplo, entre zero e um.


# In[38]:


rfm_df_scaled = pd.DataFrame(rfm_df_scaled, columns = [rfm_df.columns])
rfm_df_scaled.describe()


# In[39]:


# Rescaling the attributes

rfm_df = rfm[['ticket_medio', 'frequencia','recencia']]

# Instantiate
scaler = StandardScaler()
# Padronize recursos removendo a média e dimensionando para a variação da unidade.
# media = 0 , desvio padrao = 1

# fit_transform
rfm_df_scaled = scaler.fit_transform(rfm_df)
rfm_df_scaled.shape


# In[40]:


rfm_df_scaled = pd.DataFrame(rfm_df_scaled, columns = [rfm_df.columns])
rfm_df_scaled.describe()


# In[41]:


sns.histplot(data=rfm_df_scaled['frequencia'])


# In[42]:


import numpy as np
import seaborn as sns

#make this example reproducible
np.random.seed(0)

#create data
x = np.random.normal(size=1000)

#create normal distribution histogram
sns.displot(x, kde=True)


# <a id="4"></a> <br>
# ## Step 4 : Building the Model

# ### K-Means Clustering

# K-means clustering is one of the simplest and popular unsupervised machine learning algorithms.<br>
# 
# The algorithm works as follows:
# 
# - First we initialize k points, called means, randomly.
# - We categorize each item to its closest mean and we update the mean’s coordinates, which are the averages of the items categorized in that mean so far.
# - We repeat the process for a given number of iterations and at the end, we have our clusters.

# ### Finding the Optimal Number of Clusters

# #### Elbow Curve to get the right number of Clusters
# A fundamental step for any unsupervised algorithm is to determine the optimal number of clusters into which the data may be clustered. The Elbow Method is one of the most popular methods to determine this optimal value of k.

# In[43]:


# Elbow-curve/SSD

ssd = []
range_n_clusters = [2, 3, 4, 5, 6, 7, 8]
for num_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=num_clusters, max_iter=50)
    kmeans.fit(rfm_df_scaled)

    ssd.append(kmeans.inertia_)

# plot the SSDs for each n_clusters
plt.plot(ssd)


# ### Silhouette Analysis
# 
# $$\text{silhouette score}=\frac{p-q}{max(p,q)}$$
# 
# $p$ is the mean distance to the points in the nearest cluster that the data point is not a part of
# 
# $q$ is the mean intra-cluster distance to all the points in its own cluster.
# 
# * The value of the silhouette score range lies between -1 to 1.
# 
# * A score closer to 1 indicates that the data point is very similar to other data points in the cluster,
# 
# * A score closer to -1 indicates that the data point is not similar to the data points in its cluster.

# In[44]:


# Silhouette analysis
range_n_clusters = [2, 3, 4, 5, 6, 7, 8]

for num_clusters in range_n_clusters:

    # intialise kmeans
    kmeans = KMeans(n_clusters=num_clusters, max_iter=50)
    kmeans.fit(rfm_df_scaled)

    cluster_labels = kmeans.labels_

    # silhouette score
    silhouette_avg = silhouette_score(rfm_df_scaled, cluster_labels)
    print("For n_clusters={0}, the silhouette score is {1}".format(num_clusters, silhouette_avg))


# In[45]:


# Final model with k=4
kmeans = KMeans(n_clusters=4, max_iter=10)
kmeans.labels_= kmeans.fit_predict(rfm_df_scaled)
df_kmeans=rfm


# In[46]:


kmeans.labels_


# In[47]:


# assign the label
df_kmeans['Cluster_Id'] = kmeans.labels_.astype('str')
df_kmeans.head()


# In[48]:


# PCA não se adequa a essa base de dados
from sklearn.decomposition import PCA

pca=PCA(n_components=2)
pc=pca.fit_transform(rfm_df_scaled)

pdf=pd.DataFrame(data=pc,columns=['p1','p2'])
pdf['labels']=df_kmeans['Cluster_Id']
sns.scatterplot(data=pdf,x='p1',y='p2',hue='labels')
pdf.head()


# In[49]:


rfm.groupby('Cluster_Id').count()


# # Avaliando Kmeans

# In[50]:


#Plota comparação entre as variáveis para análise do Cluster

f, axes = plt.subplots(1, 3,figsize=(15, 8))
sns.scatterplot(data=rfm,x='frequencia',y='recencia',hue='Cluster_Id',ax=axes[0],palette="deep")
sns.scatterplot(data=rfm,x='ticket_medio',y='frequencia',hue='Cluster_Id',ax=axes[1],palette="deep")
sns.scatterplot(data=rfm,x='ticket_medio',y='recencia',hue='Cluster_Id',ax=axes[2],palette="deep")


# In[51]:


# Box plot para visualizar Cluster Id vs Frequency

sns.boxplot(x='Cluster_Id', y='ticket_medio', data=rfm)


# In[52]:


# Box plot para visualizar Cluster Id vs Frequencia

sns.boxplot(x='Cluster_Id', y='frequencia', data=rfm)


# In[53]:


# Box plot para visualizar Cluster Id vs Recência

sns.boxplot(x='Cluster_Id', y='recencia', data=rfm)


# ### Hierarchical Clustering
# 
# Hierarchical clustering involves creating clusters that have a predetermined ordering from top to bottom. For example, all files and folders on the hard disk are organized in a hierarchy. There are two types of hierarchical clustering,
# - Divisive
# - Agglomerative.

# In[54]:


plt.figure(figsize=(10, 7))
plt.title("Dendrograma")
dend = shc.dendrogram(shc.linkage(rfm_df_scaled, method='ward'))
plt.axhline(y=40, color='r', linestyle='--')


# #### Cutting the Dendrogram based on K

# In[55]:


cluster = AgglomerativeClustering(n_clusters=4)
cluster_labels = cluster.fit_predict(rfm_df_scaled)
rfm['Cluster_Labels'] = cluster_labels
hierarq = rfm
rfm.head()


# In[56]:


# Plot Cluster Id vs Frequency

sns.boxplot(x='Cluster_Labels', y='frequencia', data=rfm)


# In[57]:


# Plot Cluster Id vs Recency

sns.boxplot(x='Cluster_Labels', y='recencia', data=rfm)


# In[58]:


#para visualizar melhor essa distribuição, podemos cruzar algumas variáveis e ver como fica a clusterização
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(rfm_df_scaled)

principalDf = pd.DataFrame(data = principalComponents
             , columns = ['P1', 'P2'])
principalDf['labels']=rfm['Cluster_Labels']
sns.scatterplot(data=principalDf,x='P1',y='P2',hue='labels')


# In[59]:


#Plota comparação entre as variáveis para análise do Cluster

f, axes = plt.subplots(1, 3,figsize=(15, 8))
sns.scatterplot(data=rfm,x='frequencia',y='recencia',hue='Cluster_Labels',ax=axes[0],palette="deep")
sns.scatterplot(data=rfm,x='ticket_medio',y='frequencia',hue='Cluster_Labels',ax=axes[1],palette="deep")
sns.scatterplot(data=rfm,x='ticket_medio',y='recencia',hue='Cluster_Labels',ax=axes[2],palette="deep")


# ### DBScan

# In[60]:


from sklearn.neighbors import NearestNeighbors
neigh = NearestNeighbors(n_neighbors=5)
nbrs = neigh.fit(rfm_df_scaled)
distances, indices = nbrs.kneighbors(rfm_df_scaled)
distances = np.sort(distances, axis=0)
distances = distances[:,1]
plt.figure(figsize=(10,5))
plt.plot(distances)
plt.title('Distânica entre pontos',fontsize=20)
plt.xlabel('Pontos separados por distância',fontsize=14)
plt.ylabel('Epsilon',fontsize=14)
plt.show()


# In[61]:


db = DBSCAN(eps=0.4, min_samples=10).fit(rfm_df_scaled)
labels_dbscan = db.labels_
df_dbscan=rfm
df_dbscan['labels_dbscan']=labels_dbscan.astype('str')


# In[62]:


#Ele não consegue criar clusters
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(rfm_df_scaled)

principalDf = pd.DataFrame(data = principalComponents
             , columns = ['P1', 'P2'])
principalDf['labels']=df_dbscan['labels_dbscan']
sns.scatterplot(data=principalDf,x='P1',y='P2',hue='labels')


# ## Comparativa entre resultados

# In[63]:


# Plot Cluster Id vs Amount
f, axes = plt.subplots(1, 3,figsize=(15, 8))
sns.scatterplot(data=df_kmeans,x='frequencia',y='recencia',hue='Cluster_Id',ax=axes[0],palette="deep")
sns.scatterplot(data=hierarq,x='frequencia',y='recencia',hue='Cluster_Labels',ax=axes[1],palette="deep")
sns.scatterplot(data=rfm,x='frequencia',y='recencia',hue='labels_dbscan',ax=axes[2],palette="deep")


# In[64]:


# Plot Cluster Id vs Amount
f, axes = plt.subplots(1, 3,figsize=(15, 8))
sns.scatterplot(data=df_kmeans,x='frequencia',y='ticket_medio',hue='Cluster_Id',ax=axes[0],palette="deep")
sns.scatterplot(data=hierarq,x='frequencia',y='ticket_medio',hue='Cluster_Labels',ax=axes[1],palette="deep")
sns.scatterplot(data=rfm,x='frequencia',y='ticket_medio',hue='labels_dbscan',ax=axes[2],palette="deep")


# In[65]:


# Plot Cluster Id vs Amount
f, axes = plt.subplots(1, 3,figsize=(18, 6))
sns.boxplot(x='Cluster_Labels', y='recencia', data=rfm,ax=axes[0])
sns.boxplot(x='Cluster_Id', y='recencia', data=rfm,ax=axes[1])
sns.boxplot(x='labels_dbscan', y='recencia', data=rfm,ax=axes[2])


# In[66]:


# Plot Cluster Id vs Amount
f, axes = plt.subplots(1, 3,figsize=(18, 6))
sns.boxplot(x='Cluster_Labels', y='frequencia', data=rfm,ax=axes[0])
sns.boxplot(x='Cluster_Id', y='frequencia', data=rfm,ax=axes[1])
sns.boxplot(x='labels_dbscan', y='frequencia', data=rfm,ax=axes[2])


# <a id="5"></a> <br>
# ## Step 5 : Final Analysis

# Kmeans com 2 clusters
# - Cliente no Cluster 1 tem alta frequência e uma baixa recência e um alto ticket, significando que ele compra bastante e atualmente consome. Sugestão: Promoções e pontos para ele seguir cliente.
# - Clientes do cluster 0 já possuem um ticket menor e uma boa parte possui recencia alta e pouca frequencia. Uma sugestão seria fidelizar esse cliente através de promoções.
# 
# Algoritmo bom para escalar, caso o cliente deseje manter esse modelo.

# Cluster Hierárquico com 2 clusters
# - Cliente no Cluster 1 tem alta frequência e uma baixa recência e um alto ticket, significando que ele compra bastante e atualmente consome. Sugestão: Promoções e pontos para ele seguir cliente.
# - Clientes do cluster 0 já possuem um ticket menor e uma boa parte possui recencia alta e pouca frequencia. Uma sugestão seria fidelizar esse cliente através de promoções.
# 
# Algoritmo fácil e intuitivo, porém temos problema em escalar para grandes volumes de dados.

# DBSCAN
# Algoritmo não se adaptou a modelagem a ao tipo de base, com isso não foi útil para análises.


## Contato

#**Nome:** Lucas Oliveira Alves
#**Email:** [alves_lucasoliveira@usp.br](mailto:alves_lucasoliveira@usp.br)
#**LinkedIn:** [linkedin.com/in/cyblx](https://www.linkedin.com/in/cyblx/)
#**GitHub:** [github.com/cyblx](https://github.com/cyblx)


