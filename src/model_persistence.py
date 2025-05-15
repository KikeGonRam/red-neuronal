import numpy as np
import os

def save_model(model, filename):
    """Guarda los pesos y sesgos del modelo."""
    np.savez(filename, weights=model.weights, biases=model.biases, layers=model.layers)
    print(f"Modelo guardado en {filename}")

def load_model(filename):
    """Carga un modelo desde un archivo."""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"El archivo {filename} no existe")
    data = np.load(filename, allow_pickle=True)
    return data['weights'], data['biases'], data['layers']