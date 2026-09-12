import torch
from torch import nn


class TimeSeriesServingModel(nn.Module):
    """
    Production inference wrapper.

    Input:
        Raw time-series values
        shape: [batch, sequence_length, 1]

    Output:
        Prediction in original/raw units
        shape: [batch]
    """

    def __init__(
        self,
        model: nn.Module,
        mean: float,
        std: float,
    ):
        super().__init__()

        self.model = model

        # These are model state, but they are not trainable
        # parameters.
        #
        # register_buffer() ensures they:
        # - travel with the model state
        # - participate in serialization
        # - move with model.to(device)
        # - are NOT optimized by the optimizer
        self.register_buffer(
            "mean",
            torch.tensor(
                mean,
                dtype=torch.float32,
            ),
        )

        self.register_buffer(
            "std",
            torch.tensor(
                std,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):

        normalized_x = (
            x - self.mean
        ) / self.std

        normalized_prediction = (
            self.model(normalized_x)
        )

        prediction = (
            normalized_prediction
            * self.std
            + self.mean
        )

        return prediction
