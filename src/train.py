# src/train.py

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import pickle

from src.config import (
    DATA_FRACTION,
    NUM_SPLITS,
    BATCH_SIZE,
    LEARNING_RATE,
    NUM_EPOCHS,
    END_FACTOR,
    DEVICE,
    SAVED_MODELS_DIR,
    LOSS_HISTORY_DIR
)
from src.dataset import load_dataset, generate_splits, ArrhythmiaDataset
from src.model import LearnedFilters
from src.utils import calculate_metrics
import torch.nn as nn
import torch.optim as optim

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

def train_model(model, optimizer, scheduler, criterion, dataloader, device):
    """
    Train the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    for data, target in dataloader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        # Compute power spectra
        powerspectrum = compute_power_spectra(data.cpu().numpy())
        powerspectrum = torch.tensor(powerspectrum, dtype=torch.float32).to(device)
        data = data.unsqueeze(1)  # Add channel dimension
        output = model(data, powerspectrum)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        scheduler.step()
        running_loss += loss.item() * data.size(0)
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

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
        print(f"\n=== Split {split + 1}/{NUM_SPLITS} ===")
        train_idx, test_idx = splits[split]
        X_train, Y_train = X[train_idx], Y[train_idx]
        X_test, Y_test = X[test_idx], Y[test_idx]
        
        # Shuffle training data
        p = np.random.permutation(len(X_train))
        X_train, Y_train = X_train[p], Y_train[p]
        
        # Use data fraction
        X_train = X_train[:int(DATA_FRACTION * len(X_train))]
        Y_train = Y_train[:int(DATA_FRACTION * len(Y_train))]
        
        # Compute class weights
        class_counts = np.bincount(Y_train)
        class_weights = 1. / class_counts
        class_weights = class_weights / class_weights.sum()
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
        
        # Create DataLoader
        train_dataset = ArrhythmiaDataset(X_train, Y_train)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        # Initialize the model
        model = LearnedFilters(num_kernels=128, num_classes=3).to(DEVICE)
        
        # Define optimizer, scheduler, and loss function
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=END_FACTOR, total_iters=NUM_EPOCHS*len(train_loader))
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Training loop
        loss_history = []
        for epoch in range(NUM_EPOCHS):
            print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
            epoch_loss = train_model(model, optimizer, scheduler, criterion, train_loader, DEVICE)
            loss_history.append(epoch_loss)
            if not False:  # USE_TQDM is handled internally
                print(f"Epoch Loss: {epoch_loss:.4f}")
        
        # Append the trained model and loss history
        models.append(model)
        loss_histories.append(loss_history)
        
        # Save the trained model
        model_path = os.path.join(SAVED_MODELS_DIR, f"model_split_{split+1}.pt")
        torch.save(model.state_dict(), model_path)
        print(f"Saved model to {model_path}")
        
        # Save the loss history
        loss_path = os.path.join(LOSS_HISTORY_DIR, f"loss_split_{split+1}.pkl")
        with open(loss_path, 'wb') as f:
            pickle.dump(loss_history, f)
        print(f"Saved loss history to {loss_path}")
    
    # Save splits for later use
    splits_path = os.path.join(SAVED_MODELS_DIR, "splits.pkl")
    with open(splits_path, 'wb') as f:
        pickle.dump(splits, f)
    print(f"Saved splits to {splits_path}")

if __name__ == "__main__":
    main()
