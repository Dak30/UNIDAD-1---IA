from sklearn.cluster import KMeans
import numpy as np

X = np.array([[1, 2], [2, 3], [3, 1], [8, 8], [9, 9], [10, 10]])

kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X)
print("Centroides:", kmeans.cluster_centers_)
print("Etiquetas asignadas:", kmeans.labels_)