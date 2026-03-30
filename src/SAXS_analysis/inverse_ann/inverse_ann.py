"""MLP mapping scattering features → scaled parameters (inverse surrogate)."""

import torch.nn as nn


class InverseNN(nn.Module):
    """Fully connected network: curve features → parameter vector.

    Mirrors :class:`~SAXS_analysis.forward_ann.forward_ann.ForwardNN` layout; used to propose
    parameters from data for initialization or priors.
    """

    def __init__(self, input_size, output_size, hidden_size=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        return self.model(x)
