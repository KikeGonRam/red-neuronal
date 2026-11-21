# Red Neuronal desde cero

Red neuronal multicapa implementada **desde cero en Python con NumPy**, sin frameworks de *deep learning*. Entrena sobre el dataset Iris para clasificación multiclase y grafica las curvas de aprendizaje.

[![Seguridad](https://github.com/KikeGonRam/red-neuronal/actions/workflows/security.yml/badge.svg)](https://github.com/KikeGonRam/red-neuronal/actions/workflows/security.yml)
![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-2.2-013243?logo=numpy&logoColor=white)

[Seguridad](SECURITY.md)

## Qué incluye

| Área | Implementación |
| --- | --- |
| Arquitectura | Número de capas y neuronas configurable. |
| Activaciones | Sigmoide, ReLU y tanh (ocultas); softmax (salida). |
| Entrenamiento | Descenso de gradiente por *mini-batch* con optimizador Adam. |
| Regularización | Penalización L2 y *dropout*. |
| Métricas | Precisión, F1-score y matriz de confusión. |
| Persistencia | Guardado y carga de modelos entrenados. |
| Visualización | Curvas de pérdida, precisión y F1 con Matplotlib. |

## Estructura

```
red-neuronal/
├── src/
│   ├── neural_network.py    # Clase NeuralNetwork: forward, backward y train
│   ├── utils.py             # Activaciones y optimizador Adam
│   ├── metrics.py           # accuracy, f1 y confusion_matrix
│   └── model_persistence.py # save_model / load_model
├── data/dataset.py          # Carga y preprocesamiento del dataset Iris
├── main.py                  # Entrena, evalúa y grafica
└── test_utils.py            # Pruebas de las funciones auxiliares
```

## Instalación

```bash
git clone https://github.com/KikeGonRam/red-neuronal.git
cd red-neuronal
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Uso

```bash
python main.py
```

Entrena la red sobre Iris, imprime las métricas de prueba y abre las curvas de aprendizaje.

### API

```python
from src.neural_network import NeuralNetwork
from data.dataset import load_iris_data

X_train, X_test, y_train, y_test = load_iris_data()

nn = NeuralNetwork(
    layers=[4, 16, 8, 3],       # 4 entradas · 2 capas ocultas (16, 8) · 3 clases
    activation='relu', output_activation='softmax',
    l2_lambda=0.01, dropout_rate=0.2,
)
history = nn.train(X_train, y_train, epochs=200, batch_size=16,
                   learning_rate=0.001, optimizer='adam')
```

## Pruebas

```bash
python test_utils.py
```

## Dataset

[Iris](https://archive.ics.uci.edu/dataset/53/iris): 150 muestras, 4 características y 3 clases, cargado con scikit-learn.

## Seguridad

Escaneo de secretos (Gitleaks) y análisis estático (Semgrep) corren en cada *push* y PR
(ver [workflow](.github/workflows/security.yml)). Reporta vulnerabilidades según
[SECURITY.md](SECURITY.md).

## Tecnologías

Python · NumPy · scikit-learn · Matplotlib

---

Proyecto de Luis Enrique González Ramírez — [github.com/KikeGonRam](https://github.com/KikeGonRam)
