# grpo/block_policy.py

import torch
import torch.nn as nn


class BlockPolicy(nn.Module):
    """
    Policy that outputs gating logits per FEATURE BLOCK.

    Input:  X [B, T, F]  (sequence with regimes already in features)
    Output: logits [B, num_blocks]
    """

    def __init__(self, input_dim: int, num_blocks: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_blocks),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, F]
        returns logits: [B, num_blocks]
        """
        # Simple temporal pooling: last time step + mean
        last = x[:, -1, :]          # [B, F]
        mean = x.mean(dim=1)        # [B, F]
        pooled = torch.cat([last, mean], dim=-1)  # [B, 2F]
        logits = self.net(pooled)   # [B, num_blocks]
        return logits
