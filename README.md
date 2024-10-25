# AtrialFibrillation-Classification-SMoLK

## Overview

**AtrialFibrillation-Classification-SMoLK** is a deep learning project designed to classify arrhythmias, specifically focusing on atrial fibrillation (Afib), using the MIT-BIH Arrhythmia Database and SMoLK (Sparse Mixture of Learned Kernels). The project leverages a neural network model with learned filters to achieve high classification accuracy across multiple classes.

## Features

- **Data Loading & Preprocessing**: Efficient handling of the MIT-BIH dataset with resampling and window extraction.
- **Neural Network Model**: Implementation of the SMoLK (Sparse Mixture of Learned Kernels) model with multiple convolutional layers.
- **Training & Evaluation**: Comprehensive training scripts with cross-validation and holdout set evaluation.
- **Metrics Calculation**: Detailed metrics including Sensitivity, Specificity, AUC, and F1 Score.
- **Reproducibility**: Seed setting for consistent results across runs.

## Installation

### Prerequisites

- Python 3.8 or higher
- [Conda](https://docs.conda.io/en/latest/) (optional but recommended)

### Clone the Repository

```
git clone https://github.com/abdulvahapmutlu/atrial-fibrillation-smolk.git
cd atrial-fibrillation-smolk
```

### Using Conda
```
conda create -n afib-classification python=3.8
conda activate afib-classification
pip install -r requirements.txt
```

### Using pip
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Update Configuration:

Open src/config.py and set the DATASET_PATH to point to the extracted dataset directory.
 ``` 
DATASET_PATH = "/path/to/mit-bih-database"
```

### Configuration
All configurable parameters such as hyperparameters, file paths, and other settings are defined in src/config.py. Adjust these settings as needed for your experiments.

## Usage

### Training the Model
To train the model, execute the training script using the provided shell script:

```
bash scripts/run_training.sh
```
Alternatively, you can run the training script directly:
```
python src/train.py
```
### Evaluating the Model
After training, evaluate the model using the evaluation script:

```
bash scripts/run_evaluation.sh
```
Alternatively, run the evaluation script directly:
```
python src/evaluate.py
```
## Results

### Cross-Validation Results
Class	Sensitivity	Specificity	AUC
Normal	0.939	0.957	0.988
Afib	0.869	0.965	0.972
Other	0.947	0.977	0.993
F1 Score: 0.825 ± 0.165

### Holdout Set Results
Class	Sensitivity	Specificity	AUC
Normal	0.939	0.967	0.991
Afib	0.917	0.965	0.984
Other	0.955	0.978	0.995
F1 Score: 0.832 ± 0.160

## License
This project is licensed under the MIT License.
