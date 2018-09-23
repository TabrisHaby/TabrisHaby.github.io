---
layout:     post
title:      Clustering with KMeans and PCA
subtitle:   Example on Wine data
date:       2018-08-08
author:     Haby
header-img: img/post-bg-2015.jpg
catalog: true
tags:
    - Python
    - My Work
    - PCA
    - K-Means
    - Clustering
    - Segmentation
    - Customer
---

<br>This Wine data set contains the results of a chemical analysis of wines grown in a specific area of Italy. I use KMeans Algorithm to cluster different Wine and check if the result is correct by comparing with label variable.

@author: Ha



1.Import package and data


```python
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

import os
print(os.listdir("../input"))
```

    ['Wine.csv']



```python
df = pd.read_csv("../input/Wine.csv")
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Alcohol</th>
      <th>Malic_Acid</th>
      <th>Ash</th>
      <th>Ash_Alcanity</th>
      <th>Magnesium</th>
      <th>Total_Phenols</th>
      <th>Flavanoids</th>
      <th>Nonflavanoid_Phenols</th>
      <th>Proanthocyanins</th>
      <th>Color_Intensity</th>
      <th>Hue</th>
      <th>OD280</th>
      <th>Proline</th>
      <th>Customer_Segment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>14.23</td>
      <td>1.71</td>
      <td>2.43</td>
      <td>15.6</td>
      <td>127</td>
      <td>2.80</td>
      <td>3.06</td>
      <td>0.28</td>
      <td>2.29</td>
      <td>5.64</td>
      <td>1.04</td>
      <td>3.92</td>
      <td>1065</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>13.20</td>
      <td>1.78</td>
      <td>2.14</td>
      <td>11.2</td>
      <td>100</td>
      <td>2.65</td>
      <td>2.76</td>
      <td>0.26</td>
      <td>1.28</td>
      <td>4.38</td>
      <td>1.05</td>
      <td>3.40</td>
      <td>1050</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>13.16</td>
      <td>2.36</td>
      <td>2.67</td>
      <td>18.6</td>
      <td>101</td>
      <td>2.80</td>
      <td>3.24</td>
      <td>0.30</td>
      <td>2.81</td>
      <td>5.68</td>
      <td>1.03</td>
      <td>3.17</td>
      <td>1185</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>14.37</td>
      <td>1.95</td>
      <td>2.50</td>
      <td>16.8</td>
      <td>113</td>
      <td>3.85</td>
      <td>3.49</td>
      <td>0.24</td>
      <td>2.18</td>
      <td>7.80</td>
      <td>0.86</td>
      <td>3.45</td>
      <td>1480</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>13.24</td>
      <td>2.59</td>
      <td>2.87</td>
      <td>21.0</td>
      <td>118</td>
      <td>2.80</td>
      <td>2.69</td>
      <td>0.39</td>
      <td>1.82</td>
      <td>4.32</td>
      <td>1.04</td>
      <td>2.93</td>
      <td>735</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



2.Basic Data Information


```python
# drop Customer_Segment
label = df.Customer_Segment
df = df.drop("Customer_Segment",axis = 1)
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Alcohol</th>
      <th>Malic_Acid</th>
      <th>Ash</th>
      <th>Ash_Alcanity</th>
      <th>Magnesium</th>
      <th>Total_Phenols</th>
      <th>Flavanoids</th>
      <th>Nonflavanoid_Phenols</th>
      <th>Proanthocyanins</th>
      <th>Color_Intensity</th>
      <th>Hue</th>
      <th>OD280</th>
      <th>Proline</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>13.000618</td>
      <td>2.336348</td>
      <td>2.366517</td>
      <td>19.494944</td>
      <td>99.741573</td>
      <td>2.295112</td>
      <td>2.029270</td>
      <td>0.361854</td>
      <td>1.590899</td>
      <td>5.058090</td>
      <td>0.957449</td>
      <td>2.611685</td>
      <td>746.893258</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.811827</td>
      <td>1.117146</td>
      <td>0.274344</td>
      <td>3.339564</td>
      <td>14.282484</td>
      <td>0.625851</td>
      <td>0.998859</td>
      <td>0.124453</td>
      <td>0.572359</td>
      <td>2.318286</td>
      <td>0.228572</td>
      <td>0.709990</td>
      <td>314.907474</td>
    </tr>
    <tr>
      <th>min</th>
      <td>11.030000</td>
      <td>0.740000</td>
      <td>1.360000</td>
      <td>10.600000</td>
      <td>70.000000</td>
      <td>0.980000</td>
      <td>0.340000</td>
      <td>0.130000</td>
      <td>0.410000</td>
      <td>1.280000</td>
      <td>0.480000</td>
      <td>1.270000</td>
      <td>278.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>12.362500</td>
      <td>1.602500</td>
      <td>2.210000</td>
      <td>17.200000</td>
      <td>88.000000</td>
      <td>1.742500</td>
      <td>1.205000</td>
      <td>0.270000</td>
      <td>1.250000</td>
      <td>3.220000</td>
      <td>0.782500</td>
      <td>1.937500</td>
      <td>500.500000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>13.050000</td>
      <td>1.865000</td>
      <td>2.360000</td>
      <td>19.500000</td>
      <td>98.000000</td>
      <td>2.355000</td>
      <td>2.135000</td>
      <td>0.340000</td>
      <td>1.555000</td>
      <td>4.690000</td>
      <td>0.965000</td>
      <td>2.780000</td>
      <td>673.500000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>13.677500</td>
      <td>3.082500</td>
      <td>2.557500</td>
      <td>21.500000</td>
      <td>107.000000</td>
      <td>2.800000</td>
      <td>2.875000</td>
      <td>0.437500</td>
      <td>1.950000</td>
      <td>6.200000</td>
      <td>1.120000</td>
      <td>3.170000</td>
      <td>985.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>14.830000</td>
      <td>5.800000</td>
      <td>3.230000</td>
      <td>30.000000</td>
      <td>162.000000</td>
      <td>3.880000</td>
      <td>5.080000</td>
      <td>0.660000</td>
      <td>3.580000</td>
      <td>13.000000</td>
      <td>1.710000</td>
      <td>4.000000</td>
      <td>1680.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# checking NA
df.isnull().sum()
```




    Alcohol                 0
    Malic_Acid              0
    Ash                     0
    Ash_Alcanity            0
    Magnesium               0
    Total_Phenols           0
    Flavanoids              0
    Nonflavanoid_Phenols    0
    Proanthocyanins         0
    Color_Intensity         0
    Hue                     0
    OD280                   0
    Proline                 0
    dtype: int64



3.Data Analysis

Skewness and Outliers and Correlation


```python
def plot_multi_variable(row,column) :
    """
    Plot multi variables histogram for dataframe with row = row and colunm = column
    """
    _,ax = plt.subplots(row,column,figsize = (12,14))
    for i in range(0,row) :
        for j in range(0,column) :
            if (row*i + j >= df.shape[1]) :
                break
            else :
                sns.distplot(df.iloc[:,row*i+j],color = "red",bins = 20,ax = ax[i,j])

plot_multi_variable(4,4)
```


![png](/img/kp_1.png)



```python
def plot_boxplot_multi_variable(row,column) :
    """
    Plot multi variables boxplot for dataframe with row = row and colunm = column
    """
    _,ax = plt.subplots(row,column,figsize = (12,14))
    for i in range(0,row) :
        for j in range(0,column) :
            if (row*i + j >= df.shape[1]) :
                break
            else :
                sns.boxplot(y = df.iloc[:,row*i+j],color = "lightblue",ax = ax[i,j])

plot_boxplot_multi_variable(4,4)
```


![png](/img/kp_2.png)



```python
_,_= plt.subplots(1,1,figsize = (10,10))
corr = df.corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(corr,mask = mask,cmap=plt.cm.PuOr,square = True,annot = True)
```




    <matplotlib.axes_subplots.AxesSubplot at 0x7f8ce7eddbe0>




![png](/img/kp_3.png)


Flavanoids vs Total_Phenols : 0.86

OD280 vs Flavanoids : 0.79

OD280 vs Total_Phenols : 0.70

To large prob, these 3 variables are not I.I.D


```python
_,ax = plt.subplots(1,3,figsize = (18,3))
sns.regplot(df["Flavanoids"],df["Total_Phenols"],ax = ax[0],color = "red")
sns.regplot(df["Flavanoids"],df["OD280"],ax = ax[1],color = "green")
sns.regplot(df["Total_Phenols"],df["OD280"],ax = ax[2],color = "blue")
```


    <matplotlib.axes_subplots.AxesSubplot at 0x7f8cebdcb048>



![png](/img/kp_4.png)


4.Clustering

Kmeans is sensitive to Outliers and skewness.

Try to scale first


```python
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans

pl = Pipeline([
    ("scale",StandardScaler()),
    ("Kmeans", KMeans(n_clusters = 2,random_state = 13))
])

kmeans = pl.fit(df)
```


```python
# centroids
print(kmeans.steps[1][1].cluster_centers_)
# withinss, Total within-cluster sum of squares , larger the better
print(kmeans.steps[1][1].inertia_)
```

    [[-0.31148001  0.33837268 -0.0499309   0.46976489 -0.3074597  -0.75037054
      -0.789532    0.56770273 -0.61153123  0.0982258  -0.5400717  -0.68516469
      -0.58021779]
     [ 0.32580094 -0.35393004  0.05222657 -0.49136328  0.32159578  0.78487033
       0.82583232 -0.59380401  0.63964761 -0.10274193  0.56490258  0.71666652
       0.60689447]]
    1658.7588524290954



```python
# decide how many clusters is better
withinss = []
for i in range(1,10) :
    pl = Pipeline([
    ("scale",StandardScaler()),
    ("Kmeans", KMeans(n_clusters = i,random_state = 13))
    ])
    kmeans = pl.fit(df)
    withinss.append(kmeans.steps[1][1].inertia_)
plt.plot(withinss)
plt.ylabel("Total within-cluster sum of squares")
plt.xlabel("Number of Cluster")
```


![png](/img/kp_5.png)


K = 3 is the optimized solution based on elbow criterion

When k = 3, cluster-inside sse is 1277 < 1658 when k = 2


```python
pl_opt = Pipeline([
    ("scale",StandardScaler()),
    ("Kmeans", KMeans(n_clusters = 3,random_state = 13))
])

kmeans_opt = pl_opt.fit(df)
print(kmeans_opt.steps[1][1].inertia_)
```

    1277.9284888446423




```python
df["Cluster"] = kmeans_opt.steps[1][1].labels_
```


```python
# Centroids
centroids = kmeans_opt.steps[1][1].cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='r', zorder=10)
```




    <matplotlib.collections.PathCollection at 0x7f8cd9e50f60>




![png](/img/kp_6.png)



```python
g = sns.PairGrid(df,hue = "Cluster",hue_kws={"marker": ["o", "s", "D"]},vars = df.columns[:df.shape[1]-1])
g = g.map_diag(sns.kdeplot)
g = g.map_upper(plt.scatter)
g = g.map_lower(sns.kdeplot)
```


![png](/img/kp_7.png)


PCA analysis


```python
from sklearn.decomposition import PCA
pipeline_pca = Pipeline([
    ("scale" , StandardScaler()),
    ("PCA" ,PCA(n_components = 13,random_state = 13))
])
pca_opt = pipeline_pca.fit_transform(df.drop("Cluster",axis = 1))
```

```python
color = ["red" if i == 0 else "blue" if i == 1  else "yellow" for i in df["Cluster"]]
plt.figure(figsize = (7,7))
plt.scatter(pca_opt[:,0],pca_opt[:,2], c= color, alpha=0.8)
plt.title("PCA Decomposition with 3 Cluster")
plt.show()
```


![png](/img/kp_8.png)


5.Error Check


```python
from sklearn.metrics import classification_report
label_new = ["A" if i == 1 else "B" if i == 2 else "C" for i in label]
Cluster_new = ["A" if i == 2 else "B" if i == 0 else "C" for i in df["Cluster"]]
print(classification_report(Cluster_new,label_new))
```

                  precision    recall  f1-score   support

               A       1.00      0.95      0.98        62
               B       0.92      1.00      0.96        65
               C       1.00      0.94      0.97        51

       micro avg       0.97      0.97      0.97       178
       macro avg       0.97      0.96      0.97       178
    weighted avg       0.97      0.97      0.97       178




```python
# Accuracy
Correct = [1 if label_new[i] == Cluster_new[i] else 0 for i in range(0,len(label_new))]
print("Total Accuracy is :",np.mean(Correct))
```

    Total Accuracy is : 0.9662921348314607
