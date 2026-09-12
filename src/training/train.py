from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.dataset import TimeSeriesDataset
from src.data.prepare import load_and_split
from src.models.lstm import LSTMForecaster


SEED = 42

DATA_PATH = "data/raw/timeseries.csv"

SEQUENCE_LENGTH = 24
BATCH_SIZE = 256

HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.1

LEARNING_RATE = 1e-3
EPOCHS = 10

CHECKPOINT_PATH = Path(
    "artifacts/checkpoints/lstm_best.pt"
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_loader(
    values,
    shuffle: bool,
    pin_memory: bool,
):
    dataset = TimeSeriesDataset(
        values,
        sequence_length=SEQUENCE_LENGTH,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=pin_memory,
    )

    return loader


def evaluate(
    model,
    loader,
    criterion,
    device,
):
    model.eval()

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():

        for x, y in loader:

            x = x.to(
                device,
                non_blocking=True,
            )

            y = y.to(
                device,
                non_blocking=True,
            )

            prediction = model(x)

            loss = criterion(
                prediction,
                y,
            )

            batch_size = x.size(0)

            total_loss += (
                loss.item()
                * batch_size
            )

            total_samples += batch_size

    return total_loss / total_samples


def main():

    set_seed(SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    pin_memory = (
        device.type == "cuda"
    )

    print("Device:", device)

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    splits = load_and_split(
        DATA_PATH
    )

    print()
    print(
        "Train observations:",
        len(splits.train),
    )
    print(
        "Validation observations:",
        len(splits.validation),
    )
    print(
        "Test observations:",
        len(splits.test),
    )

    print(
        f"Training mean: {splits.mean:.4f}"
    )
    print(
        f"Training std:  {splits.std:.4f}"
    )

    train_loader = create_loader(
        splits.train,
        shuffle=True,
        pin_memory=pin_memory,
    )

    validation_loader = create_loader(
        splits.validation,
        shuffle=False,
        pin_memory=pin_memory,
    )

    model = LSTMForecaster(
        input_size=1,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)

    criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    print()
    print(
        "Model parameters:",
        parameter_count,
    )

    CHECKPOINT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_validation_loss = float("inf")

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        model.train()

        running_loss = 0.0
        samples_seen = 0

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        for x, y in train_loader:

            x = x.to(
                device,
                non_blocking=True,
            )

            y = y.to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True,
            )

            prediction = model(x)

            loss = criterion(
                prediction,
                y,
            )

            loss.backward()

            optimizer.step()

            batch_size = x.size(0)

            running_loss += (
                loss.item()
                * batch_size
            )

            samples_seen += batch_size

        train_loss = (
            running_loss
            / samples_seen
        )

        validation_loss = evaluate(
            model=model,
            loader=validation_loader,
            criterion=criterion,
            device=device,
        )

        if device.type == "cuda":
            peak_memory_mb = (
                torch.cuda.max_memory_allocated()
                / 1024**2
            )
        else:
            peak_memory_mb = 0.0

        print(
            f"Epoch {epoch:02d}/{EPOCHS} "
            f"train_loss={train_loss:.6f} "
            f"val_loss={validation_loss:.6f} "
            f"peak_gpu_mem={peak_memory_mb:.1f} MiB"
        )

        if (
            validation_loss
            < best_validation_loss
        ):

            best_validation_loss = (
                validation_loss
            )

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),
                    "optimizer_state_dict":
                        optimizer.state_dict(),
                    "validation_loss":
                        validation_loss,
                    "epoch":
                        epoch,
                    "mean":
                        splits.mean,
                    "std":
                        splits.std,
                    "sequence_length":
                        SEQUENCE_LENGTH,
                    "hidden_size":
                        HIDDEN_SIZE,
                    "num_layers":
                        NUM_LAYERS,
                },
                CHECKPOINT_PATH,
            )

    print()
    print(
        "Best validation loss:",
        f"{best_validation_loss:.6f}",
    )

    print(
        "Checkpoint:",
        CHECKPOINT_PATH,
    )


if __name__ == "__main__":
    main()
