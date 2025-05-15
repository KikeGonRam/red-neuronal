import numpy as np
from src.neural_network import NeuralNetwork
from data.dataset import generate_data, plot_data

def main():
    # Generar datos
    X, y = generate_data(n_samples=200)
    plot_data(X, y, title="Dataset Sintético")
    
    # Crear red neuronal [2 entradas, 4 neuronas ocultas, 1 salida]
    nn = NeuralNetwork([2, 4, 1])
    
    # Entrenar
    nn.train(X, y, epochs=1000, learning_rate=0.1)
    
    # Evaluar
    output = nn.forward(X)
    predictions = (output > 0.5).astype(int)
    accuracy = np.mean(predictions == y)
    print(f"Precisión: {accuracy:.4f}")
    
    # Visualizar predicciones
    plot_data(X, predictions.flatten(), title="Predicciones de la Red Neuronal")

if __name__ == "__main__":
    main()