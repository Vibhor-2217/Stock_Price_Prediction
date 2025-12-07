import torch
import torch.nn as nn


class TwoHeadLSTM(nn.Module):
    """
    Two-head LSTM inspired by base_lstm_2:
      - One head predicts next-day return (regression).
      - One head predicts direction (logit for up/down).
    """
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Regression head: hidden -> hidden/2 -> 1
        reg_hidden = hidden_dim // 2
        self.reg_head = nn.Sequential(
            nn.Linear(hidden_dim, reg_hidden),
            nn.ReLU(),
            nn.Linear(reg_hidden, 1),
        )

        # Classification head: hidden -> hidden/2 -> 1 (logit)
        cls_hidden = hidden_dim // 2
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_dim, cls_hidden),
            nn.ReLU(),
            nn.Linear(cls_hidden, 1),
        )

    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        lstm_out, _ = self.lstm(x)
        last = lstm_out[:, -1, :]  # [batch, hidden_dim]

        ret_pred = self.reg_head(last).squeeze(-1)  # [batch]
        dir_logit = self.cls_head(last).squeeze(-1)  # [batch] (for BCEWithLogits)

        return dir_logit, ret_pred
