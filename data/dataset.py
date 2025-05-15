import numpy as np
import matplotlib.pyplot as plt

def generate_data(n_samples=100):
    """Genera un dataset sintético de dos clases."""
    np.random.seed(42)
    
    # Clase 0: Puntos alrededor de (0, 0)
    class0 = np.random.randn(n_samples // 2, 2) * 0.5
    
    # Clase 1: Puntos alrededor de (2, 2)
    class1 = np.random.randn(n_samples // 2, 2) * 0.5 + 2
    
    X = np.vstack([class0, class1])
    y = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2)).reshape(-1, 1)
    
    return X, y

def plot_data(X, y, title="Dataset"):
    """Visualiza el dataset."""
    plt.scatter(X[y.flatten() == 0][:, 0], X[y.flatten() == 0][:, 1], label="Clase 0")
    plt.scatter(X[y.flatten() == 1][:, 0], X[y.flatten() == 1][:, 1], label="Clase 1")
    plt.title(title)
    plt.legend()
    plt.show()