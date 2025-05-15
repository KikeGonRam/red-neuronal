Proyecto de Red Neuronal Avanzada
Este proyecto implementa una red neuronal avanzada desde cero en Python usando NumPy. Soporta múltiples capas, funciones de activación (sigmoide, ReLU, tanh), mini-batch gradient descent, regularización (L2, dropout), optimizador Adam, métricas avanzadas, y guardado/carga de modelos. Usa el dataset Iris para clasificación multiclase.
Estructura

src/: Código fuente.
neural_network.py: Implementación de la red neuronal.
utils.py: Funciones de activación y optimizadores.
metrics.py: Cálculo de métricas (precisión, F1-score).
model_persistence.py: Guardado y carga de modelos.


data/: Carga y preprocesamiento del dataset Iris.
main.py: Script principal para entrenar, evaluar y visualizar.
requirements.txt: Dependencias.

Instalación
pip install -r requirements.txt

Uso
python main.py

Dataset
Usa el dataset Iris para clasificación multiclase (3 clases).
Características

Mini-batch gradient descent.
Regularización L2 y dropout.
Optimizador Adam.
Métricas: precisión, F1-score.
Guardado/carga de modelos.

