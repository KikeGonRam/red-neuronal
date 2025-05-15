import numpy as np

def sigmoid(x):
    """Función de activación sigmoide."""
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """Derivada de la sigmoide."""
    sig = sigmoid(x)
    return sig * (1 - sig)

def relu(x):
    """Función de activación ReLU."""
    return np.maximum(0, x)

def relu_derivative(x):
    """Derivada de ReLU."""
    return np.where(x > 0, 1, 0)

def tanh(x):
    """Función de activación tanh."""
    return np.tanh(x)

def tanh_derivative(x):
    """Derivada de tanh."""
    return 1 - np.tanh(x) ** 2

def softmax(x):
    """Función de activación softmax."""
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def get_activation(name):
    """Devuelve la función de activación y su derivada según el nombre."""
    activations = {
        'sigmoid': (sigmoid, sigmoid_derivative),
        'relu': (relu, relu_derivative),
        'tanh': (tanh, tanh_derivative),
        'softmax': (softmax, None)  # Softmax no necesita derivada en este caso
    }
    return activations.get(name, (sigmoid, sigmoid_derivative))

def adam_optimizer(params, gradients, m, v, t, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
    """Actualiza parámetros usando el optimizador Adam."""
    m = [beta1 * m_i + (1 - beta1) * grad for m_i, grad in zip(m, gradients)]
    v = [beta2 * v_i + (1 - beta2) * (grad ** 2) for v_i, grad in zip(v, gradients)]
    m_hat = [m_i / (1 - beta1 ** t) for m_i in m]
    v_hat = [v_i / (1 - beta2 ** t) for v_i in v]
    params = [p - learning_rate * m_h / (np.sqrt(v_h) + epsilon) for p, m_h, v_h in zip(params, m_hat, v_hat)]
    return params, m, v