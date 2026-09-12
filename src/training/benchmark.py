import argparse
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.dataset import TimeSeriesDataset
from src.data.prepare import load_and_split
from src.models.lstm import LSTMForecaster


DATA_PATH = "data/raw/timeseries.csv"

SEQUENCE_LENGTH = 24
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.1
LEARNING_RATE = 1e-3
SEED = 42


def create_loader(
    values,
    batch_size,
    shuffle,
    pin_memory,
):
    dataset = TimeSeriesDataset(
        values,
        sequence_length=SEQUENCE_LENGTH,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=pin_memory,
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--precision",
        choices=["fp32", "amp"],
        default="fp32",
    )

    args = parser.parse_args()

    if (
        args.device == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA requested but unavailable"
        )

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device(args.device)

    use_amp = (
        args.precision == "amp"
        and device.type == "cuda"
    )

    if args.precision == "amp" and device.type != "cuda":
        raise RuntimeError(
            "AMP benchmark currently requires CUDA"
        )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
    )

    splits = load_and_split(
        DATA_PATH
    )

    train_loader = create_loader(
        splits.train,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=(
            device.type == "cuda"
        ),
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

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    print("=== BENCHMARK CONFIG ===")
    print("Device:", device)

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    print(
        "Batch size:",
        args.batch_size,
    )

    print(
        "Epochs:",
        args.epochs,
    )

    print(
        "Precision:",
        args.precision,
    )

    print()

    total_samples = 0

    training_start = time.perf_counter()

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        epoch_start = time.perf_counter()

        model.train()

        epoch_samples = 0
        running_loss = 0.0

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

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                prediction = model(x)

                loss = criterion(
                    prediction,
                    y,
                )

            scaler.scale(loss).backward()

            scaler.step(optimizer)

            scaler.update()

            batch_samples = x.size(0)

            epoch_samples += (
                batch_samples
            )

            running_loss += (
                loss.item()
                * batch_samples
            )

        if device.type == "cuda":
            torch.cuda.synchronize()

        epoch_time = (
            time.perf_counter()
            - epoch_start
        )

        samples_per_second = (
            epoch_samples
            / epoch_time
        )

        average_loss = (
            running_loss
            / epoch_samples
        )

        total_samples += (
            epoch_samples
        )

        print(
            f"epoch={epoch:02d} "
            f"loss={average_loss:.6f} "
            f"time={epoch_time:.3f}s "
            f"throughput="
            f"{samples_per_second:.1f} samples/s"
        )

    if device.type == "cuda":
        torch.cuda.synchronize()

    total_time = (
        time.perf_counter()
        - training_start
    )

    overall_throughput = (
        total_samples
        / total_time
    )

    print()
    print("=== BENCHMARK RESULT ===")

    print(
        f"Total time: "
        f"{total_time:.3f}s"
    )

    print(
        f"Overall throughput: "
        f"{overall_throughput:.1f} samples/s"
    )

    if device.type == "cuda":

        peak_memory = (
            torch.cuda.max_memory_allocated()
            / 1024**2
        )

        print(
            f"Peak GPU memory: "
            f"{peak_memory:.1f} MiB"
        )


if __name__ == "__main__":
    main()
