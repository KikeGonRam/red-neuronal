# Red Neuronal desde cero

Red neuronal multicapa implementada **desde cero en Python con NumPy**, sin frameworks de deep learning. Entrena sobre el dataset Iris para clasificación multiclase y muestra las curvas de aprendizaje.

![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-2.2-013243?logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-F7931E?logo=scikitlearn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.10-11557C)

## Características

- **Arquitectura configurable:** número de capas y neuronas por capa.
- **Funciones de activación:** sigmoide, ReLU y tanh en capas ocultas; softmax en la salida.
- **Entrenamiento:** descenso de gradiente por *mini-batch* con optimizador **Adam**.
- **Regularización:** penalización **L2** y **dropout**.
- **Métricas:** precisión, F1-score y matriz de confusión.
- **Persistencia:** guardado y carga de modelos entrenados.
- **Visualización:** curvas de pérdida, precisión y F1 por época con Matplotlib.

## Estructura

```
red-neuronal/
├── src/
│   ├── neural_network.py   # Clase NeuralNetwork: forward, backward y train
│   ├── utils.py            # Activaciones y optimizador Adam
│   ├── metrics.py          # accuracy, f1 y confusion_matrix
│   └── model_persistence.py# save_model / load_model
├── data/
│   └── dataset.py          # Carga y preprocesamiento del dataset Iris
├── main.py                 # Entrena, evalúa y grafica las curvas
├── test_utils.py           # Pruebas de las funciones auxiliares
└── requirements.txt
```

## Instalación

```bash
git clone https://github.com/KikeGonRam/red-neuronal.git
cd red-neuronal
python -m venv .venv && source .venv/bin/activate   # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Uso

```bash
python main.py
```

Entrena la red sobre Iris, imprime las métricas en el conjunto de prueba y abre las curvas de aprendizaje.

### Ejemplo de la API

```python
from src.neural_network import NeuralNetwork
from data.dataset import load_iris_data

X_train, X_test, y_train, y_test = load_iris_data()

nn = NeuralNetwork(
    layers=[4, 16, 8, 3],       # 4 entradas · 2 capas ocultas (16, 8) · 3 clases
    activation='relu',
    output_activation='softmax',
    l2_lambda=0.01,
    dropout_rate=0.2,
)

history = nn.train(
    X_train, y_train,
    epochs=200,
    batch_size=16,
    learning_rate=0.001,
    optimizer='adam',
)
```

## Dataset

[Iris](https://archive.ics.uci.edu/dataset/53/iris): 150 muestras, 4 características y 3 clases. Se carga con scikit-learn y se divide en entrenamiento y prueba.

## Pruebas

```bash
python test_utils.py
```

## Tecnologías

Python · NumPy · scikit-learn (carga del dataset) · Matplotlib

---

Proyecto educativo de Luis Enrique González Ramírez — [github.com/KikeGonRam](https://github.com/KikeGonRam)
