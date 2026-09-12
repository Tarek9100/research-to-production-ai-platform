import torch
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    def __init__(
        self,
        values,
        sequence_length: int,
    ):
        self.values = torch.tensor(
            values,
            dtype=torch.float32,
        )

        self.sequence_length = sequence_length

    def __len__(self):
        return (
            len(self.values)
            - self.sequence_length
        )

    def __getitem__(self, idx):
        x = self.values[
            idx: idx + self.sequence_length
        ]

        y = self.values[
            idx + self.sequence_length
        ]

        # Add feature dimension.
        x = x.unsqueeze(-1)

        return x, y
