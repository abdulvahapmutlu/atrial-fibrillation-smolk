# src/dataset.py

import os
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.signal import periodogram, resample
import wfdb
from tqdm import tqdm
import pickle

from src.config import (
    DATASET_PATH,
    SAMPLE_RATE,
    TARGET_SAMPLE_RATE,
    WINDOW_SIZE,
    USE_TQDM
)

def load_record(record_name, path, target_fs=TARGET_SAMPLE_RATE):
    """
    Load a single record, resample, and extract R-peaks and annotations.
    """
    record = wfdb.rdrecord(os.path.join(path, record_name))
    annotation = wfdb.rdann(os.path.join(path, record_name), 'atr')
    
    # Select first channel (usually MLII)
    signal = record.p_signal[:,0]
    
    # Resample signal
    num_samples = int(len(signal) * target_fs / SAMPLE_RATE)
    signal_resampled = resample(signal, num_samples)
    
    # Adjust R-peak locations after resampling
    r_peaks = (annotation.sample * target_fs) // SAMPLE_RATE
    r_peaks = r_peaks[r_peaks < len(signal_resampled)]
    
    # Extract annotations
    annotations = annotation.symbol
    return signal_resampled, r_peaks, annotations

def extract_windows(signal, r_peaks, annotations, window_size=WINDOW_SIZE):
    """
    Extract windows around R-peaks and assign labels.
    """
    half_window = window_size // 2
    windows = []
    labels = []
    
    for peak, symbol in zip(r_peaks, annotations):
        start = peak - half_window
        end = peak + half_window
        if start < 0 or end > len(signal):
            continue  # Skip if window is out of bounds
        window = signal[start:end]
        windows.append(window)
        labels.append(symbol)
    return np.array(windows), np.array(labels)

def map_labels(labels):
    """
    Map original MIT-BIH labels to desired classes:
    - "Normal" (N, L, R, e, j)
    - "Afib" (A, a, F, J, S)
    - "Other" (all other classes)
    """
    normal = ['N', 'L', 'R', 'e', 'j']  # Including some variants
    afib = ['A', 'a', 'F', 'J', 'S']
    mapped_labels = []
    for label in labels:
        if label in normal:
            mapped_labels.append(0)  # Normal
        elif label in afib:
            mapped_labels.append(1)  # Afib
        else:
            mapped_labels.append(2)  # Other
    return np.array(mapped_labels)

def load_dataset(path=DATASET_PATH):
    """
    Load all records, extract windows and labels.
    """
    # List of all record names in the dataset
    records = [f.split('.')[0] for f in os.listdir(path) if f.endswith('.dat')]
    
    all_windows = []
    all_labels = []
    
    for record in tqdm(records, desc="Loading Records", disable=not USE_TQDM):
        signal, r_peaks, annotations = load_record(record, path)
        windows, labels = extract_windows(signal, r_peaks, annotations)
        windows = windows[:, :WINDOW_SIZE]  # Ensure consistent window size
        all_windows.append(windows)
        all_labels.append(labels)
    
    all_windows = np.concatenate(all_windows, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    mapped_labels = map_labels(all_labels)
    
    # Exclude "Other" class if needed
    # For demonstration, we'll keep all classes
    return all_windows, mapped_labels

def generate_splits(X, Y, num_splits=10):
    """
    Generate cross-validation splits.
    """
    splits = []
    unique_classes = np.unique(Y)
    for split in range(num_splits):
        # Simple random split; for better stratification, implement stratified splits
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        train_size = int(0.8 * len(X))
        train_idx = indices[:train_size]
        test_idx = indices[train_size:]
        splits.append((train_idx, test_idx))
    return splits

class ArrhythmiaDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.long)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
