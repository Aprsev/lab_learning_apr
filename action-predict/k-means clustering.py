import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# 生成模拟数据
n_samples = 300
n_clusters = 3
X, y = make_blobs(n_samples=n_samples, centers=n_clusters, random_state=42)

# 使用k均值聚类
kmeans = KMeans(n_clusters=n_clusters,n_init=10, max_iter=500,random_state=42)
kmeans.fit(X)

# 获取聚类结果和聚类中心
labels = kmeans.labels_
centers = kmeans.cluster_centers_

# 可视化聚类结果
plt.figure(figsize=(8, 6))

# 绘制每个簇的数据点
for cluster_label in range(n_clusters):
    cluster_points = X[labels == cluster_label]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster_label + 1}')

# 绘制聚类中心
plt.scatter(centers[:, 0], centers[:, 1], marker='x', color='k', s=100, label='Centroids')

plt.title('K-means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)
plt.show()


