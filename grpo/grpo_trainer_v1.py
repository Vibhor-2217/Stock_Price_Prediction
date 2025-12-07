import torch
import torch.nn as nn
import torch.optim as optim

from models.two_head_lstm_v2 import TwoHeadLSTM_V2
from models.state_encoder_v1 import StateEncoderV1
from models.auxiliary_heads_v1 import AuxiliaryHeadsV1, auxiliary_loss_fn


class SimpleGRPO(nn.Module):
    def __init__(self, state_dim=64, action_dim=3):
        super().__init__()

        self.policy = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

        self.value = nn.Linear(state_dim, 1)

    def forward(self, z):
        return self.policy(z), self.value(z)


# --------------------------------------------------
# TRAINING LOOP
# --------------------------------------------------

def train_one_batch(model, encoder, grpo, aux, optimizer,
                    batch_x, true_next_ret, true_regime):
    """
    batch_x: [T, B, input_dim]
    """

    # 1) LSTM
    H1, H2 = model(batch_x)

    # 2) Encode state
    z = encoder(H1, H2)     # [B, state_dim]

    # 3) GRPO forward
    logits, value = grpo(z)

    # Fake GRPO targets (placeholder)
    # You will replace these with your actual GRPO computation
    policy_loss = logits.mean()
    value_loss = value.mean()

    grpo_loss = policy_loss + 0.5 * value_loss

    # 4) Auxiliary losses
    next_pred, regime_logits = aux(z)
    aux_loss = auxiliary_loss_fn(next_pred, true_next_ret,
                                 regime_logits, true_regime)

    # 5) Total loss
    total_loss = grpo_loss + 0.1 * aux_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return total_loss.item()


def build_model(input_dim, hidden_dim, state_dim):

    model = TwoHeadLSTM_V2(input_dim, hidden_dim)
    encoder = StateEncoderV1(hidden_dim, state_dim)
    grpo = SimpleGRPO(state_dim)
    aux = AuxiliaryHeadsV1(state_dim, num_regimes = 4)

    optimizer = optim.Adam(
        list(model.parameters()) +
        list(encoder.parameters()) +
        list(grpo.parameters()) +
        list(aux.parameters()),
        lr=1e-4
    )

    return model, encoder, grpo, aux, optimizer
