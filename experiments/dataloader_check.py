import pandas as pd
from torch.utils.data import DataLoader

from src.data.dataset import TimeSeriesDataset


SEQUENCE_LENGTH = 24
BATCH_SIZE = 64

df = pd.read_csv(
    "data/raw/timeseries.csv"
)

values = df["value"].values

dataset = TimeSeriesDataset(
    values,
    sequence_length=SEQUENCE_LENGTH,
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    pin_memory=True,
)

x, y = next(iter(loader))

print("Dataset samples:", len(dataset))

print("x shape:", x.shape)
print("y shape:", y.shape)

print("x dtype:", x.dtype)
print("y dtype:", y.dtype)

print("x pinned:", x.is_pinned())

print()
print("Example sequence:")
print(x[0, :, 0])

print()
print("Target:")
print(y[0])
