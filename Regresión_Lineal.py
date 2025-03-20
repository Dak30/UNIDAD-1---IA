import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Datos de entrenamiento
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([2, 3, 5, 7, 11])

# Crear y entrenar el modelo
model = LinearRegression()
model.fit(X, y)  # Ajustar el modelo con los datos

# Predicción
X_pred = np.array([6]).reshape(-1, 1)
prediction = model.predict(X_pred)  # Hacer la predicción
print(f"Predicción para X=6: {prediction}")
