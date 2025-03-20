from sklearn.linear_model import Perceptron
import numpy as np

# Datos de entrenamiento (AND lógico)
X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([0, 0, 0, 1])  # Salida de la compuerta lógica AND

# Crear y entrenar el perceptrón
perceptron = Perceptron()
perceptron.fit(X_train, y_train)  # Ajustar el modelo con los datos

# Predicción
print(perceptron.predict([[0, 1], [1, 1]]))  # Predice para dos ejemplos
