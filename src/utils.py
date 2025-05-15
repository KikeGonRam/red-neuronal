import numpy as np

def sigmoid(x):
    """Función de activación sigmoide."""
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """Derivada de la función sigmoide."""
    sig = sigmoid(x)
    return sig * (1 - sig)