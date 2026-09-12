import torch
from torch import nn


class LSTMForecaster(nn.Module):
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        self.output = nn.Linear(
            hidden_size,
            1,
        )

    def forward(self, x):
        sequence_output, _ = self.lstm(x)

        # Keep the hidden representation from
        # the final time step.
        last_hidden = sequence_output[:, -1, :]

        prediction = self.output(last_hidden)

        # [batch, 1] -> [batch]
        return prediction.squeeze(-1)
