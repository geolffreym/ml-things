from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import normalize, LabelEncoder
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import seaborn as sns
import pandas as pd
import numpy as np

# Data read
# 1. sepal length in cm
# 2. sepal width in cm
# 3. petal length in cm
# 4. petal width in cm
# 5. class:
df = pd.read_csv('iris.data')
df.columns = ['sepal_l_cm', 'sepal_w_cm', 'petal_l_cm', 'petal_w_cm', 'class']

# Split data
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(df['class'])

X = np.array(df.drop('class', axis=1))
# Normalize data
X = normalize(X)

# K-means test
k_test = range(1, 11)
k_means_minim_error = []

# Test clusters
# Elbow graph
for k in k_test:
    k_means = KMeans(n_clusters=k)
    k_means.fit(X)
    k_means_minim_error.append(
        k_means.inertia_
    )

# plt.plot(k_test, k_means_minim_error)
# plt.show()

# K-means Clustering
k_means = KMeans(n_clusters=3, random_state=0, max_iter=500)
# Hierarchical Clustering
agg_ward = AgglomerativeClustering(n_clusters=3, linkage='ward')
agg_average = AgglomerativeClustering(n_clusters=3, linkage='average')
agg_complete = AgglomerativeClustering(n_clusters=3, linkage='complete')
dbscan = DBSCAN(eps=1, min_samples=1)

for clustering in [k_means, agg_average, agg_ward, agg_complete, dbscan]:
    # Labels predicted
    labels_pred = clustering.fit_predict(X)
    # Metric to know how well the labels were predicted based on original samples labels
    # External indices
    print("\n", adjusted_rand_score(labels, labels_pred))
    print(silhouette_score(X, labels_pred))
    # print((labels_pred == labels).sum() / labels_pred.shape[0])

# k_means.fit(X)
# labels_pred = k_means.predict(X)
#
# plt.scatter(X[:, [0]], X[:, [1]], c='b', s=50, cmap='viridis')
# centers = k_means.cluster_centers_
# plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)

# Multi dimension graph
g = sns.PairGrid(df, hue='class')
g.map(plt.scatter)

exit()
linkage_matrix = linkage(X, 'ward')
dendrogram(linkage_matrix)
plt.show()

# plt.scatter(X[labels_pred == 0, 0], X[labels_pred == 0, 1], s=100, c='green', label='VersiColor')
# plt.scatter(X[labels_pred == 1, 0], X[labels_pred == 1, 1], s=100, c='red', label='Setosa')
# plt.scatter(X[labels_pred == 2, 0], X[labels_pred == 2, 1], s=100, c='blue', label='Virginica')
# plt.scatter(k_means.cluster_centers_[:, 0], k_means.cluster_centers_[:, 1], s=100, c='orange', label='Centroid')
# plt.legend()
# plt.show()
