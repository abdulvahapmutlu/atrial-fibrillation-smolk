# src/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class LearnedFilters(nn.Module):
    """
    SMoLK Model: Learned Filters for Arrhythmia Classification

    This model applies multiple convolutional filters of varying kernel sizes
    to the input signal, extracts features by averaging the activations, and
    combines them with power spectrum features for classification.
    """
    def __init__(self, num_kernels=128, num_classes=3):
        super(LearnedFilters, self).__init__()
        self.conv1 = nn.Conv1d(1, num_kernels, 192, stride=1, bias=True)
        self.conv2 = nn.Conv1d(1, num_kernels, 96, stride=1, bias=True)
        self.conv3 = nn.Conv1d(1, num_kernels, 64, stride=1, bias=True)
        self.linear = nn.Linear(num_kernels*3 + 321, num_classes)  # 321 is the size of the power spectrum
    
    def forward(self, x, powerspectrum):
        c1 = F.leaky_relu(self.conv1(x)).mean(dim=-1)
        c2 = F.leaky_relu(self.conv2(x)).mean(dim=-1)
        c3 = F.leaky_relu(self.conv3(x)).mean(dim=-1)
        aggregate = torch.cat([c1, c2, c3, powerspectrum], dim=1)
        aggregate = self.linear(aggregate)
        return aggregate
