import numpy as np
from sklearn.metrics import recall_score, f1_score, confusion_matrix

def gmean_score(y_true, y_pred) -> float:
    """G-mean = sqrt(TPR * TNR)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    return float(np.sqrt(tpr * tnr))

def compute_metrics(y_true, y_pred) -> dict:
    return {
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "gmean": gmean_score(y_true, y_pred),
    }
