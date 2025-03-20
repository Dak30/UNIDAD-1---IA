from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# Datos de ejemplo
X_train = np.array([[1, 2], [2, 3], [3, 1], [5, 4], [6, 5]])
y_train = np.array([0, 0, 0, 1, 1])

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Predicción para un nuevo punto
new_point = np.array([[4, 3]])
pred = knn.predict(new_point)
print(f"Clase predicha: {pred}")