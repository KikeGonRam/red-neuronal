import numpy as np
from .utils import sigmoid, sigmoid_derivative

class NeuralNetwork:
    def __init__(self, layers):
        """Inicializa la red neuronal.
        Args:
            layers: Lista con el número de neuronas por capa (ej. [2, 3, 1]).
        """
        self.layers = layers
        self.weights = []
        self.biases = []
        self.activations = []
        
        # Inicializar pesos y sesgos
        for i in range(len(layers) - 1):
            w = np.random.randn(layers[i], layers[i + 1]) * 0.01
            b = np.zeros((1, layers[i + 1]))
            self.weights.append(w)
            self.biases.append(b)
    
    def forward(self, X):
        """Propagación hacia adelante."""
        self.activations = [X]
        current = X
        
        for i in range(len(self.weights)):
            z = np.dot(current, self.weights[i]) + self.biases[i]
            current = sigmoid(z)
            self.activations.append(current)
        
        return current
    
    def backward(self, X, y, learning_rate):
        """Propagación hacia atrás."""
        m = X.shape[0]
        delta = self.activations[-1] - y  # Error en la salida
        
        for i in range(len(self.weights) - 1, -1, -1):
            dW = np.dot(self.activations[i].T, delta) / m
            db = np.sum(delta, axis=0, keepdims=True) / m
            
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * sigmoid_derivative(self.activations[i])
            
            # Actualizar pesos y sesgos
            self.weights[i] -= learning_rate * dW
            self.biases[i] -= learning_rate * db
    
    def train(self, X, y, epochs, learning_rate):
        """Entrena la red neuronal."""
        for epoch in range(epochs):
            # Propagación hacia adelante
            output = self.forward(X)
            
            # Propagación hacia atrás
            self.backward(X, y, learning_rate)
            
            # Calcular y mostrar pérdida (error cuadrático medio)
            loss = np.mean((output - y) ** 2)
            if epoch % 100 == 0:
                print(f"Epoca {epoch}, Pérdida: {loss:.4f}")