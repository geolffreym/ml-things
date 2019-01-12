DB_HOST = '54.202.25.197'
DB_DATABASE = 'envios2224'
DB_USER = 'geo'
DB_PASS = 'ebf6b08d5bcca2365f9627eb10715155'

from sklearn.cluster import KMeans

from app.agency.raw_sql import RAW_QUERY
# from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import pandas as pd
import pymysql
import numpy as np

from matplotlib.patches import Ellipse


def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()

    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)

    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))


def plot_gmm(gmm, X, label=True, ax=None):
    ax = ax or plt.gca()
    labels = gmm.fit(X).predict(X)
    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)
    ax.axis('equal')

    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)


def fetch_from_query(query_index, db=DB_DATABASE, **kwargs):
    # ################SETTINGS MYSQL##############
    # Connecting to mysql
    db = pymysql.connect(
        charset='utf8',
        use_unicode=True,
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASS,
        db=db
    )

    # Db cursor
    cursor = db.cursor()
    # Execute query
    # cursor.execute(query_index)
    cursor.execute(RAW_QUERY[query_index].format(**kwargs))
    # Order from db
    return cursor.fetchall()


data = fetch_from_query('acivities_cluster_weekend')
df = pd.DataFrame(
    list(data),
    columns=['status', 'agency_id', 'agency_name', 'created_at']
)

df.created_at = pd.to_datetime(df['created_at'])

paid = df[df.status == 4]
sent = df[df.status == 3]

# Group by day
paid_group = paid.resample('W-MON', on='created_at')
sent_group = sent.resample('W-MON', on='created_at')

cols = pd.concat([paid_group.size(), sent_group.size()], ignore_index=True, axis=1)
cols.fillna(0, inplace=True)

# cols = cols[(cols.T != 0).any()]
X = cols.values

estimator = KMeans(n_clusters=3)
estimator.fit(X)
centers = estimator.cluster_centers_
labels_pred = estimator.predict(X)

# K-means test
k_test = range(1, 11)
k_means_minim_error = []

# for k in k_test:
#     k_means = KMeans(n_clusters=k)
#     k_means.fit(cols)
#     k_means_minim_error.append(
#         k_means.inertia_
#     )

# estimator = DBSCAN(eps=2, min_samples=3)
# labels_pred = estimator.fit_predict(X)
#
# estimator = AgglomerativeClustering(linkage='ward', n_clusters=3)
# estimator.fit(X)
# label_pred = estimator.fit_predict(X)
# # linkage_matrix = linkage(cols, 'ward')
# # dendrogram(linkage_matrix)
# # plt.show()
# for k in k_test:
#     estimator = GaussianMixture(n_components=k + 1, covariance_type='full', max_iter=300, n_init=10)
#     estimator.fit(X)
#     # estimator = AgglomerativeClustering(linkage='ward', n_clusters=k + 1)
#     # labels_pred = estimator.fit_predict(X)
#     # labels_pred = estimator.predict(X)
#     # estimator = KMeans(n_clusters=k + 1)
#     # estimator.fit(X)
#     labels_pred = estimator.predict(X)
#     k_means_minim_error.append(
#         silhouette_score(X, labels_pred)
#     )

#estimator = GaussianMixture(n_components=3, max_iter=300, n_init=10)
# estimator.fit(X)
# labels_pred = estimator.predict(X)

print (silhouette_score(X, labels_pred))
print (calinski_harabaz_score(X, labels_pred))
#
# plt.plot(k_test, k_means_minim_error)
# plt.show()

colors = ['b', 'g', 'c']
markers = ['o', 'v', 's']

# for i, l in enumerate(estimator.labels_):
#     plt.plot(X[[i], [0]], X[[i], [1]], color=colors[l], marker=markers[l], ls='None')
# plt.show()

# probs = estimator.predict_proba(X)
# size = 50 * probs.max(1) ** 2  # square emphasizes differences
# # plt.scatter(X[:, 0], X[:, 1], c=labels_pred, cmap='viridis')
# plt.scatter(X[:, 0], X[:, 1], c=labels_pred, cmap='viridis', s=size)
# plt.scatter(centers[:, 0], centers[:, 1], marker="x", color='r')


plot_gmm(estimator, X)
plt.ylabel('Paid')
plt.xlabel('Sent')
labels = []

for k, v in enumerate(cols.index.values):
    b = None
    if k < len(cols.index.values)  -1:
        b = pd.to_datetime(str(cols.index.values[k + 1])).strftime('%Y.%m.%d')

    a = pd.to_datetime(str(v)).strftime('%Y.%m.%d')
    labels.append(a + (b and ' | ' + b or ''))


plt.subplots_adjust(bottom=0.1)

for label, x, y in zip(labels, X[:, 0], X[:, 1]):
    plt.annotate(
        label,
        xy=(x, y), xytext=None,
        textcoords='offset points', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.1', fc='yellow', alpha=0.8),
        arrowprops=dict(arrowstyle='->'))

plt.show()

import ipdb

ipdb.set_trace()
print(estimator.fit(X).predict([2, 3]))
