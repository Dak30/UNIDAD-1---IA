# Taller de Preparación,  Inteligencia Artificial, 2025A

## Temas Cubiertos:
1. Regresión Lineal
2. K-Nearest Neighbors (KNN)
3. K-Means Clustering
4. Análisis de Componentes Principales (PCA)
5. Perceptrón Simple
6. Redes Neuronales con 1-2 capas ocultas

## Instrucciones:
- Complete el código en los espacios indicados.
- Responda las preguntas conceptuales.
- Consulte las referencias sugeridas para reforzar tu aprendizaje.

---

### 1. Regresión Lineal
#### a) Complete el código para entrenar un modelo de regresión lineal en Python con scikit-learn:
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Datos de entrenamiento
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([2, 3, 5, 7, 11])

# Crear y entrenar el modelo
model = LinearRegression()
model.fit(X, y)

# Predicción
X_pred = np.array([6]).reshape(-1, 1)
prediction = model.predict(X_pred)
print(f"Predicción para X=6: {prediction}")
```

R:// Predicción para X=6: [12.2]

#### b) Pregunta Conceptual:
¿Qué significan los coeficientes del modelo de regresión lineal?

En un modelo de regresión lineal, los coeficientes representan la relación entre las variables independientes (X) y la variable dependiente (y).

- **Coeficiente (\(BETA \))**: Indica cuánto cambia \(y\) por cada unidad de cambio en \(X\).  
  - Si \(BETA > 0\): Relación positiva (cuando \(X\) aumenta, \(y\) también).  
  - Si \(BETA < 0\): Relación negativa (cuando \(X\) aumenta, \(y\) disminuye).  
  - Si \(BETA = 0\): No hay relación entre \(X\) y \(y\).  

- **Intercepto (\(\BETA\))**: Es el valor de \(y\) cuando \(X = 0\), es decir, dónde la línea de regresión cruza el eje \(y\).  

```

---

### 2. K-Nearest Neighbors (KNN)
#### a) Complete el código para clasificar puntos usando KNN:
```python
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
```
R// Clase predicha: [0]
#### b) Pregunta Conceptual:
¿Cómo afecta el valor de `k` al modelo?

R//
###  Efecto del valor de `k` en KNN  
El valor de `k` en **K-Nearest Neighbors (KNN)** controla cuántos vecinos se consideran para clasificar un punto nuevo:  

- **Valores pequeños de `k` (ej. 1, 3)**:  
  - El modelo es más sensible a los datos individuales (puede sobreajustar).  
  - Mayor riesgo de ruido y varianza alta.  

- **Valores grandes de `k` (ej. 10, 20)**:  
  - Se suaviza la clasificación, reduciendo el impacto del ruido.  
  - Puede generar subajuste si `k` es demasiado grande.  

 Un buen `k` se elige mediante validación cruzada para equilibrar sesgo y varianza.  

---

### 3. K-Means Clustering
#### a) Complete el código para realizar clustering con K-Means:
```python
from sklearn.cluster import KMeans
import numpy as np

X = np.array([[1, 2], [2, 3], [3, 1], [8, 8], [9, 9], [10, 10]])

kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X)
print("Centroides:", kmeans.cluster_centers_)
print("Etiquetas asignadas:", kmeans.labels_)
```
R//
Centroides: [[2. 2.]
 [9. 9.]]
Etiquetas asignadas: [0 0 0 1 1 1]

#### b) Pregunta Conceptual:
¿Qué significa el número de clusters en K-Means?
El número de clusters (`n_clusters`) en **K-Means** define cuántos grupos se formarán en los datos.  

- Cada cluster tiene un **centroide**, que representa el punto medio del grupo.  
- K-Means asigna cada dato al cluster cuyo centroide esté más cercano.  
- Un número muy bajo de clusters puede **agrupar datos distintos** en el mismo grupo (subajuste).  
- Un número muy alto de clusters puede **dividir datos similares** en grupos innecesarios (sobreajuste).  

---

### 4. Análisis de Componentes Principales (PCA)
#### a) Complete el código para reducir la dimensionalidad con PCA:
```python
from sklearn.decomposition import PCA
import numpy as np

X = np.array([[2.5, 2.4], [0.5, 0.7], [2.2, 2.9], [1.9, 2.2], [3.1, 3.0]])

pca = PCA(n_components=1)
X_reduced = pca.fit_transform(X)
print("Datos reducidos:", X_reduced)
```
R//: 
Datos reducidos: [[ 0.44362444]
 [-2.17719404]
 [ 0.57071239]
 [-0.12902465]
 [ 1.29188186]]
 
#### b) Pregunta Conceptual:
¿Cómo se interpretan los componentes principales?

Los **componentes principales** en **PCA** son nuevas variables que capturan la mayor variabilidad de los datos originales.  

- **Cada componente principal** es una combinación lineal de las variables originales.  
- **El primer componente (PC1)** explica la mayor cantidad de varianza posible.  
- **El segundo componente (PC2)** es ortogonal a PC1 y explica la siguiente mayor varianza, y así sucesivamente.  
- Los **valores propios** indican cuánta varianza explica cada componente.  
- Los **vectores propios** muestran la contribución de cada variable en cada componente. 

---

### 5. Perceptrón Simple
#### a) Complete el código para entrenar un perceptrón en Python:
```python
from sklearn.linear_model import Perceptron
import numpy as np

X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([0, 0, 0, 1])  # AND lógico

perceptron = Perceptron()
perceptron.fit(X_train, y_train)

# Predicción
print(perceptron.predict([[0, 1], [1, 1]]))
```
R//: [0 1]

#### b) Pregunta Conceptual:
¿Por qué el perceptrón no puede resolver el problema XOR?

El perceptrón simple **no puede resolver XOR** porque **no es linealmente separable**.  

- **El perceptrón solo aprende fronteras de decisión lineales**, pero XOR necesita una **frontera no lineal**.  
- En XOR, no hay una línea recta que pueda separar correctamente las clases 0 y 1 en el espacio de entrada.  
- Se necesita al menos **una capa oculta** (como en redes neuronales multicapa) para capturar la relación no lineal.  

---

### 6. Redes Neuronales con 1-2 capas ocultas
#### a) Complete el código para una red neuronal con una capa oculta en Keras:
```python
import tensorflow as tf
from tensorflow import keras

# Definir la red neuronal con una capa oculta
model = keras.Sequential([
    keras.layers.Dense(4, activation='relu', input_shape=(2,)),  # Capa oculta con 4 neuronas y activación ReLU
    keras.layers.Dense(1, activation='sigmoid')  # Capa de salida con activación sigmoide
])

# Compilar el modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

```

#### b) Pregunta Conceptual:
¿Qué efecto tiene aumentar el número de neuronas en una capa oculta?

### ¿Qué efecto tiene aumentar el número de neuronas en una capa oculta?

Aumentar el número de neuronas en una capa oculta puede mejorar la capacidad de la red para aprender patrones más complejos, ya que permite modelar funciones más detalladas y relaciones no lineales en los datos. Sin embargo, también puede aumentar el riesgo de **sobreajuste** (overfitting), haciendo que el modelo memorice en lugar de generalizar. Para evitarlo, es recomendable usar técnicas como **regularización (L1/L2), dropout o más datos de entrenamiento**.

---

## Referencias
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [TensorFlow & Keras](https://www.tensorflow.org/)
- [Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course)

---