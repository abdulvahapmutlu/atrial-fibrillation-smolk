# src/utils.py

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score

def calculate_metrics(y_true, y_prob, num_classes=3):
    """
    Calculate sensitivity, specificity, AUC, and F1 scores.
    """
    y_pred = np.argmax(y_prob, axis=1)
    sensitivities = []
    specificities = []
    AUCs = []
    F1s = []
    
    for cls in range(num_classes):
        # Binary labels for the current class
        true_binary = (y_true == cls).astype(int)
        pred_binary = (y_pred == cls).astype(int)
        
        # Compute confusion matrix
        cm = confusion_matrix(true_binary, pred_binary).ravel()
        if len(cm) == 4:
            tn, fp, fn, tp = cm
        elif len(cm) == 2:
            tn, tp = cm
            fp, fn = 0, 0
        elif len(cm) == 1:
            tp = cm[0]
            tn, fp, fn = 0, 0, 0
        else:
            tn, fp, fn, tp = 0, 0, 0, 0
        
        # Sensitivity (Recall)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivities.append(sensitivity)
        specificities.append(specificity)
        
        # AUC
        try:
            auc = roc_auc_score(true_binary, y_prob[:, cls])
        except ValueError:
            auc = 0  # If only one class is present in y_true, AUC is not defined
        AUCs.append(auc)
        
        # F1 Score
        f1 = f1_score(true_binary, pred_binary, zero_division=0)
        F1s.append(f1)
    
    return sensitivities, specificities, AUCs, F1s

def print_table(sensitivities, specificities, AUCs, class_names):
    """
    Print a table of metrics.
    """
    print(f"{'Class':<10}{'Sensitivity':<15}{'Specificity':<15}{'AUC':<10}")
    for i, class_name in enumerate(class_names):
        print(f"{class_name:<10}{sensitivities[i]:<15.3f}{specificities[i]:<15.3f}{AUCs[i]:<10.3f}")
