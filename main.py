import numpy as np
import matplotlib.pyplot as plt
from src.neural_network import NeuralNetwork
from src.metrics import accuracy, f1, confusion_matrix
from src.model_persistence import save_model, load_model
from data.dataset import load_iris_data

def plot_learning_curves(history):
    """Grafica las curvas de aprendizaje."""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(history['loss'], label='Pérdida')
    plt.title('Pérdida')
    plt.xlabel('Época')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(history['accuracy'], label='Precisión')
    plt.title('Precisión')
    plt.xlabel('Época')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(history['f1'], label='F1-score')
    plt.title('F1-score')
    plt.xlabel('Época')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def main():
    # Cargar datos
    X_train, X_test, y_train, y_test = load_iris_data()
    
    # Crear red neuronal
    nn = NeuralNetwork(
        layers=[4, 16, 8, 3],  # 4 entradas, 2 capas ocultas (16, 8), 3 salidas
        activation='relu',
        output_activation='softmax',
        l2_lambda=0.01,
        dropout_rate=0.2
    )
    
    # Entrenar
    history = nn.train(
        X_train, y_train,
        epochs=200,
        batch_size=16,
        learning_rate=0.001,
        optimizer='adam'
    )
    
    # Evaluar en conjunto de prueba
    output = nn.forward(X_test, training=False)
    predictions = np.argmax(output, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    acc = accuracy(y_true, predictions)
    f1_val = f1(y_true, predictions)
    cm = confusion_matrix(y_true, predictions, num_classes=3)
    
    print(f"Precisión en prueba: {acc:.4f}")
    print(f"F1-score en prueba: {f1_val:.4f}")
    print("Matriz de confusión:")
    print(cm)
    
    # Graficar curvas de aprendizaje
    plot_learning_curves(history)
    
    # Guardar modelo
    save_model(nn, 'model.npz')
    
    # Cargar modelo (demostración)
    weights, biases, layers = load_model('model.npz')
    nn_loaded = NeuralNetwork(layers=layers, activation='relu', output_activation='softmax')
    nn_loaded.weights = weights
    nn_loaded.biases = biases
    output_loaded = nn_loaded.forward(X_test, training=False)
    predictions_loaded = np.argmax(output_loaded, axis=1)
    acc_loaded = accuracy(y_true, predictions_loaded)
    print(f"Precisión con modelo cargado: {acc_loaded:.4f}")

if __name__ == "__main__":
    main()