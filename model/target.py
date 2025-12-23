"""
EECS 445 - Introduction to Machine Learning
Winter 2025 - Project 2

Target CNN
    Constructs a pytorch model for a convolutional neural network
    Usage: from model.target import Target
"""

from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["Target"]


class Target(nn.Module):
    def __init__(self) -> None:
        """Define model architecture."""
        super().__init__()

        # TODO: 2(b) - define each layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=2, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=8, kernel_size=5, stride=2, padding=2)
        self.fc_1 = nn.Linear(in_features=32, out_features=2)

        self.init_weights()
        

    def init_weights(self) -> None:
        """Initialize model weights."""
        torch.manual_seed(42)
        for conv in [self.conv1, self.conv2, self.conv3]:
            # TODO: 2(b) - initialize the parameters for the convolutional layers
            calculatedSTD = sqrt(1 / (5 * 5 * conv.in_channels))
            nn.init.normal_(conv.weight, mean=0.0, std=calculatedSTD)
            nn.init.constant_(conv.bias, 0.0)
            pass

        # TODO: 2(b) - initialize the parameters for [self.fc_1]
        nn.init.normal_(self.fc_1.weight, mean=0.0, std=sqrt(1 / self.fc_1.in_features))
        nn.init.constant_(self.fc_1.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, H, W = x.shape

        # TODO: 2(b) - , forward pass
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.view(N, -1)
        x = self.fc_1(x)
        return x

