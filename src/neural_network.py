import numpy as np
from .utils import get_activation, adam_optimizer
from .metrics import f1  # Importar f1 desde metrics.py

class NeuralNetwork:
    def __init__(self, layers, activation='relu', output_activation='softmax', l2_lambda=0.01, dropout_rate=0.2):
        """Inicializa la red neuronal.
        Args:
            layers: Lista con el número de neuronas por capa.
            activation: Función de activación para capas ocultas ('sigmoid', 'relu', 'tanh').
            output_activation: Función de activación para la capa de salida.
            l2_lambda: Parámetro de regularización L2.
            dropout_rate: Tasa de dropout.
        """
        self.layers = layers
        self.activation, self.activation_deriv = get_activation(activation)
        self.output_activation, _ = get_activation(output_activation)
        self.l2_lambda = l2_lambda
        self.dropout_rate = dropout_rate
        self.weights = []
        self.biases = []
        self.activations = []
        self.dropouts = []
        
        # Inicializar pesos y sesgos
        for i in range(len(layers) - 1):
            w = np.random.randn(layers[i], layers[i + 1]) * np.sqrt(2 / layers[i])  # Inicialización He
            b = np.zeros((1, layers[i + 1]))
            self.weights.append(w)
            self.biases.append(b)
    
    def forward(self, X, training=True):
        """Propagación hacia adelante."""
        self.activations = [X]
        self.dropouts = []
        current = X
        
        for i in range(len(self.weights) - 1):
            z = np.dot(current, self.weights[i]) + self.biases[i]
            current = self.activation(z)
            if training and self.dropout_rate > 0:
                dropout_mask = np.random.binomial(1, 1 - self.dropout_rate, size=current.shape) / (1 - self.dropout_rate)
                current *= dropout_mask
                self.dropouts.append(dropout_mask)
            else:
                self.dropouts.append(np.ones_like(current))
            self.activations.append(current)
        
        # Capa de salida
        z = np.dot(current, self.weights[-1]) + self.biases[-1]
        current = self.output_activation(z)
        self.activations.append(current)
        self.dropouts.append(np.ones_like(current))
        
        return current
    
    def backward(self, X, y, learning_rate):
        """Propagación hacia atrás."""
        m = X.shape[0]
        delta = self.activations[-1] - y  # Error en la salida (para softmax + cross-entropy)
        
        gradients_w = []
        gradients_b = []
        
        for i in range(len(self.weights) - 1, -1, -1):
            dW = np.dot(self.activations[i].T, delta) / m + self.l2_lambda * self.weights[i]
            db = np.sum(delta, axis=0, keepdims=True) / m
            
            gradients_w.insert(0, dW)
            gradients_b.insert(0, db)
            
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self.activation_deriv(self.activations[i]) * self.dropouts[i-1]
        
        return gradients_w, gradients_b
    
    def train(self, X, y, epochs, batch_size=32, learning_rate=0.001, optimizer='adam'):
        """Entrena la red neuronal."""
        m = X.shape[0]
        history = {'loss': [], 'accuracy': [], 'f1': []}
        
        # Inicializar Adam
        if optimizer == 'adam':
            m_w = [np.zeros_like(w) for w in self.weights]
            v_w = [np.zeros_like(w) for w in self.weights]
            m_b = [np.zeros_like(b) for b in self.biases]
            v_b = [np.zeros_like(b) for b in self.biases]
            t = 0
        
        for epoch in range(epochs):
            # Mini-batch
            indices = np.random.permutation(m)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            for i in range(0, m, batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]
                
                # Forward
                output = self.forward(X_batch, training=True)
                
                # Backward
                grad_w, grad_b = self.backward(X_batch, y_batch, learning_rate)
                
                # Actualizar parámetros
                if optimizer == 'adam':
                    t += 1
                    params = self.weights + self.biases
                    gradients = grad_w + grad_b
                    m_params = m_w + m_b
                    v_params = v_w + v_b
                    params, m_params, v_params = adam_optimizer(params, gradients, m_params, v_params, t, learning_rate)
                    self.weights = params[:len(self.weights)]
                    self.biases = params[len(self.weights):]
                    m_w, m_b = m_params[:len(self.weights)], m_params[len(self.weights):]
                    v_w, v_b = v_params[:len(self.weights)], v_params[len(self.weights):]
                else:
                    for j in range(len(self.weights)):
                        self.weights[j] -= learning_rate * grad_w[j]
                        self.biases[j] -= learning_rate * grad_b[j]
            
            # Calcular métricas
            output = self.forward(X, training=False)
            loss = -np.mean(y * np.log(output + 1e-15))  # Cross-entropy
            predictions = np.argmax(output, axis=1)
            y_true = np.argmax(y, axis=1)
            acc = np.mean(predictions == y_true)
            f1_val = f1(y_true, predictions)  # Usar f1 de metrics.py
            
            history['loss'].append(loss)
            history['accuracy'].append(acc)
            history['f1'].append(f1_val)
            
            if epoch % 10 == 0:
                print(f"Epoca {epoch}, Pérdida: {loss:.4f}, Precisión: {acc:.4f}, F1: {f1_val:.4f}")
        
        return history