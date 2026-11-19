"""PyTorch multilayer-perceptron model used for the neural-network detector."""

import torch.nn as nn


class SpamMLP(nn.Module):
    """A small feed-forward neural network for TF-IDF text vectors."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, 1),
        )

    def forward(self, features):
        return self.network(features).squeeze(1)
