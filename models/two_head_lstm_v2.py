import torch
import torch.nn as nn


class TwoHeadLSTM_V2(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.lstm_head1 = nn.LSTM(
            input_dim, hidden_dim, batch_first=False
        )
        self.lstm_head2 = nn.LSTM(
            input_dim, hidden_dim, batch_first=False
        )

    def forward(self, x):
        """
        x: [T, batch, input_dim]
        Returns:
            H1: [T, batch, hidden_dim]
            H2: [T, batch, hidden_dim]
        """
        H1, _ = self.lstm_head1(x)
        H2, _ = self.lstm_head2(x)

        return H1, H2
