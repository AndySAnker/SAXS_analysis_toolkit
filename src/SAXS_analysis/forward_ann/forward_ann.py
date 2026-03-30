"""MLP mapping scaled parameters → scattering features (forward surrogate)."""

import torch.nn as nn


class ForwardNN(nn.Module):
    """Fully connected network: parameter vector → model output (e.g. curve features).

    Architecture: input → hidden (ReLU) ×3 → linear output. Used as a fast stand-in for
    sasmodels during training or MCMC when paired with the inverse network.
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
