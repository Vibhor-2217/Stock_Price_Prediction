import torch
import torch.nn as nn


class LSTMBaseline(nn.Module):
    def __init__(self, input_dim, hidden_dim = 64, num_layers = 2, dropout = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
                input_size = input_dim,
                hidden_size = hidden_dim,
                num_layers = num_layers,
                batch_first = True,
                dropout = dropout
        )

        # Two output heads:
        self.fc_dir = nn.Linear(hidden_dim, 2)  # for classification
        self.fc_price = nn.Linear(hidden_dim, 1)  # for regression

    def forward(self, x):
        # x: (batch, seq_len, features)
        out, (h, c) = self.lstm(x)

        # Last hidden state for prediction
        last = out[:, -1, :]

        direction_logits = self.fc_dir(last)
        price_pred = self.fc_price(last).squeeze(-1)

        return direction_logits, price_pred
