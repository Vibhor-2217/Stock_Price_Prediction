import torch
import torch.nn as nn
import torch.nn.functional as F


class StateEncoderV1(nn.Module):
    def __init__(self, hidden_dim, state_dim=64):
        super().__init__()

        # Fusion of two LSTM heads
        self.fuse = nn.Linear(hidden_dim * 2, hidden_dim)

        # Attention scorer
        self.attn = nn.Linear(hidden_dim, 1)

        # Residual projection
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, state_dim)
        )

        self.ln = nn.LayerNorm(state_dim)

    def forward(self, H1, H2):
        """
        H1, H2: [T, batch, hidden_dim]
        Output:
            z: [batch, state_dim]
        """

        # fuse heads (time major)
        H = torch.cat([H1, H2], dim=-1)     # [T, B, 2H]
        H = F.relu(self.fuse(H))            # [T, B, H]

        # attention weights across T
        scores = self.attn(H).squeeze(-1)   # [T, B]
        weights = torch.softmax(scores, dim=0)  # attention over time

        # weighted sum
        z_attn = torch.sum(weights.unsqueeze(-1) * H, dim=0)  # [B, H]

        # projection
        z_proj = self.mlp(z_attn)  # [B, state_dim]

        # residual (pad if needed)
        if z_attn.shape[1] != z_proj.shape[1]:
            pad = z_proj.shape[1] - z_attn.shape[1]
            z_attn = F.pad(z_attn, (0, pad))

        z = self.ln(z_proj + z_attn)
        return z
