import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def load_iris_data(test_size=0.2, random_state=42):
    """Carga y preprocesa el dataset Iris."""
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    # Escalar características
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Codificar etiquetas (one-hot)
    encoder = OneHotEncoder(sparse_output=False)
    y = encoder.fit_transform(y.reshape(-1, 1))
    
    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    return X_train, X_test, y_train, y_test