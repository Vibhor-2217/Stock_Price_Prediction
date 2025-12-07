# grpo/feature_policy.py

import torch
import torch.nn as nn


class FeaturePolicy(nn.Module):
    """
    Simple feature-selection policy for GRPO.

    Input:  X [B, T, F]  (sequence of features)
    Output: logits [B, F] -> per-feature Bernoulli probs via sigmoid
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, F]
        returns logits: [B, F]
        """
        # simple temporal pooling: mean over time
        pooled = x.mean(dim=1)  # [B, F]
        logits = self.fc(pooled)  # [B, F]
        return logits
