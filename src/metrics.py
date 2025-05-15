import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

def accuracy(y_true, y_pred):
    """Calcula la precisión."""
    return np.mean(y_true == y_pred)

def f1(y_true, y_pred):
    """Calcula el F1-score."""
    return f1_score(y_true, y_pred, average='macro')

def confusion_matrix(y_true, y_pred, num_classes):
    """Calcula la matriz de confusión."""
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm