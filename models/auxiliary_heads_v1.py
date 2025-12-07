import torch
import torch.nn as nn


class AuxiliaryHeadsV1(nn.Module):
    def __init__(self, state_dim=64, num_regimes=4):
        super().__init__()
        self.next_return = nn.Linear(state_dim, 1)
        self.regime_classifier = nn.Linear(state_dim, num_regimes)

    def forward(self, z):
        next_ret = self.next_return(z)         # [B, 1]
        regime_logits = self.regime_classifier(z)  # [B, C]
        return next_ret, regime_logits


def auxiliary_loss_fn(next_pred, next_true, regime_logits, regime_true):
    """
    next_pred: [B, 1]
    next_true: [B, 1]
    regime_logits: [B, C]
    regime_true: [B]
    """
    mse = nn.MSELoss()(next_pred, next_true)
    ce = nn.CrossEntropyLoss()(regime_logits, regime_true)
    return 0.5 * mse + 0.5 * ce
