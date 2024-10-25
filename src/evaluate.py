# src/evaluate.py

import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import (
    NUM_SPLITS,
    BATCH_SIZE,
    DEVICE,
    SAVED_MODELS_DIR,
    LOSS_HISTORY_DIR
)
from src.dataset import load_dataset, generate_splits, ArrhythmiaDataset
from src.model import LearnedFilters
from src.utils import calculate_metrics, print_table
import pickle

def compute_power_spectra(X_batch, target_fs=64):
    """
    Compute power spectra for a batch of samples using periodogram.
    """
    from scipy.signal import periodogram
    PowerSpectra = []
    for i in range(len(X_batch)):
        f, Pxx = periodogram(X_batch[i], fs=target_fs)
        # To match size 321, interpolate or truncate
        if len(Pxx) < 321:
            Pxx = np.pad(Pxx, (0, 321 - len(Pxx)), 'constant')
        else:
            Pxx = Pxx[:321]
        PowerSpectra.append(Pxx)
    return np.array(PowerSpectra).astype(np.float32)

def evaluate_model(model, dataloader, device):
    """
    Evaluate the model and return probabilities and ground truth.
    """
    model.eval()
    probs = []
    ground_truth = []
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            powerspectrum = compute_power_spectra(data.cpu().numpy())
            powerspectrum = torch.tensor(powerspectrum, dtype=torch.float32).to(device)
            data = data.unsqueeze(1)  # Add channel dimension
            output = model(data, powerspectrum).softmax(dim=-1)
            probs.append(output.cpu().numpy())
            ground_truth.append(target.cpu().numpy())
    probs = np.concatenate(probs, axis=0)
    ground_truth = np.concatenate(ground_truth, axis=0)
    return probs, ground_truth

def main():
    # Load and preprocess the dataset
    print("Loading and preprocessing the dataset...")
    X, Y = load_dataset()
    print(f"Total samples: {len(X)}")
    print(f"Class distribution: {np.bincount(Y)}")
    
    # Generate cross-validation splits
    splits = generate_splits(X, Y, NUM_SPLITS)
    
    models = []
    loss_histories = []
    for split in range(NUM_SPLITS):
        # Load models and loss histories if needed
        pass  # Not needed for evaluation
    
    # Cross-Validation Evaluation
    print("\n=== Cross-Validation Evaluation ===")
    sensitivities = []
    specificities = []
    AUCs = []
    F1s = []
    class_names = ["Normal", "Afib", "Other"]
    
    for split in range(NUM_SPLITS):
        print(f"\n--- Evaluating Split {split + 1} ---")
        train_idx, test_idx = splits[split]
        X_test, Y_test = X[test_idx], Y[test_idx]
        
        test_dataset = ArrhythmiaDataset(X_test, Y_test)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        model = LearnedFilters(num_kernels=128, num_classes=3).to(DEVICE)
        model_path = os.path.join(SAVED_MODELS_DIR, f"model_split_{split+1}.pt")
        model.load_state_dict(torch.load(model_path))
        model.to(DEVICE)
        
        probs, ground_truth = evaluate_model(model, test_loader, DEVICE)
        
        sen, spec, auc, f1 = calculate_metrics(ground_truth, probs, num_classes=3)
        sensitivities.append(sen)
        specificities.append(spec)
        AUCs.append(auc)
        F1s.append(f1)
    
    sensitivities = np.array(sensitivities)
    specificities = np.array(specificities)
    AUCs = np.array(AUCs)
    F1s = np.array(F1s)
    
    print("\n=== Cross-Validation Results ===")
    print_table(sensitivities.mean(axis=0), specificities.mean(axis=0), AUCs.mean(axis=0), class_names)
    print(f"F1 Score: {F1s.mean():.3f} ± {F1s.std():.3f}")
    
    # Holdout Set Evaluation
    print("\n=== Holdout Set Evaluation ===")
    # For demonstration, we'll use the last split as holdout
    holdout_split = NUM_SPLITS - 1
    train_idx, holdout_idx = splits[holdout_split]
    X_holdout, Y_holdout = X[holdout_idx], Y[holdout_idx]
    holdout_dataset = ArrhythmiaDataset(X_holdout, Y_holdout)
    holdout_loader = DataLoader(holdout_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    holdout_sensitivities = []
    holdout_specificities = []
    holdout_AUCs = []
    holdout_F1s = []
    
    for split in range(NUM_SPLITS):
        print(f"\n--- Evaluating Model {split + 1} on Holdout Set ---")
        model = LearnedFilters(num_kernels=128, num_classes=3).to(DEVICE)
        model_path = os.path.join(SAVED_MODELS_DIR, f"model_split_{split+1}.pt")
        model.load_state_dict(torch.load(model_path))
        model.to(DEVICE)
        
        probs, ground_truth = evaluate_model(model, holdout_loader, DEVICE)
        
        sen, spec, auc, f1 = calculate_metrics(ground_truth, probs, num_classes=3)
        holdout_sensitivities.append(sen)
        holdout_specificities.append(spec)
        holdout_AUCs.append(auc)
        holdout_F1s.append(f1)
    
    holdout_sensitivities = np.array(holdout_sensitivities)
    holdout_specificities = np.array(holdout_specificities)
    holdout_AUCs = np.array(holdout_AUCs)
    holdout_F1s = np.array(holdout_F1s)
    
    print("\n=== Holdout Set Results ===")
    print_table(holdout_sensitivities.mean(axis=0), holdout_specificities.mean(axis=0), holdout_AUCs.mean(axis=0), class_names)
    print(f"F1 Score: {holdout_F1s.mean():.3f} ± {holdout_F1s.std():.3f}")

if __name__ == "__main__":
    main()
