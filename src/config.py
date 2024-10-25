# src/config.py

import os

# ===========================
# Set Seed for Reproducibility
# ===========================
def set_seed(seed=0):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed(0)

# ===========================
# Configuration
# ===========================

# Path to the MIT-BIH dataset (update this path accordingly)
DATASET_PATH = "/path/to/mit-bih-arrhythmia-database-1.0.0"

# Sampling rates
SAMPLE_RATE = 360  # Original sampling rate for MIT-BIH
TARGET_SAMPLE_RATE = 64  # Downsampled rate

# Window settings
WINDOW_SIZE_SEC = 3.0  # Window size in seconds around R-peak
WINDOW_SIZE = int(WINDOW_SIZE_SEC * TARGET_SAMPLE_RATE)  # 192 samples
OVERLAP = 0  # No overlap

# Training settings
NUM_SPLITS = 10
DATA_FRACTION = 1.0
NUM_KERNELS = 128
LEARNING_RATE = 0.001
BATCH_SIZE = 256
NUM_EPOCHS = 16
END_FACTOR = 0.1
USE_TQDM = False  # Disable tqdm during training
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Directories for saving models and loss history
SAVED_MODELS_DIR = "saved_models"
LOSS_HISTORY_DIR = "loss_history"

os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
os.makedirs(LOSS_HISTORY_DIR, exist_ok=True)
