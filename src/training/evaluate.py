from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import TimeSeriesDataset
from src.data.prepare import load_and_split
from src.models.lstm import LSTMForecaster


DATA_PATH = "data/raw/timeseries.csv"

CHECKPOINT_PATH = Path(
    "artifacts/checkpoints/lstm_best.pt"
)

METRICS_PATH = Path(
    "artifacts/evaluation/test_metrics.json"
)

PLOT_PATH = Path(
    "artifacts/evaluation/test_predictions.png"
)

BATCH_SIZE = 256


def main():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    # --------------------------------------------------
    # 1. Load checkpoint
    # --------------------------------------------------

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
    )

    print(
        "Checkpoint epoch:",
        checkpoint["epoch"],
    )

    print(
        "Checkpoint validation loss:",
        checkpoint["validation_loss"],
    )

    # --------------------------------------------------
    # 2. Recreate model architecture
    # --------------------------------------------------

    model = LSTMForecaster(
        input_size=1,
        hidden_size=checkpoint["hidden_size"],
        num_layers=checkpoint["num_layers"],
        dropout=0.1,
    ).to(device)

    # --------------------------------------------------
    # 3. Restore trained parameters
    # --------------------------------------------------

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    # --------------------------------------------------
    # 4. Recreate the same data split
    # --------------------------------------------------

    splits = load_and_split(
        DATA_PATH
    )

    test_dataset = TimeSeriesDataset(
        splits.test,
        sequence_length=checkpoint[
            "sequence_length"
        ],
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    # --------------------------------------------------
    # 5. Inference
    # --------------------------------------------------

    predictions = []
    targets = []

    with torch.no_grad():

        for x, y in test_loader:

            x = x.to(
                device,
                non_blocking=True,
            )

            prediction = model(x)

            predictions.append(
                prediction.cpu().numpy()
            )

            targets.append(
                y.numpy()
            )

    predictions = np.concatenate(
        predictions
    )

    targets = np.concatenate(
        targets
    )

    # --------------------------------------------------
    # 6. Convert normalized values back to original units
    # --------------------------------------------------

    mean = checkpoint["mean"]
    std = checkpoint["std"]

    predictions_original = (
        predictions * std + mean
    )

    targets_original = (
        targets * std + mean
    )

    # --------------------------------------------------
    # 7. Calculate metrics
    # --------------------------------------------------

    errors = (
        predictions_original
        - targets_original
    )

    mse = float(
        np.mean(errors ** 2)
    )

    rmse = float(
        np.sqrt(mse)
    )

    mae = float(
        np.mean(
            np.abs(errors)
        )
    )

    metrics = {
        "checkpoint_epoch":
            checkpoint["epoch"],

        "validation_loss":
            checkpoint["validation_loss"],

        "test_mse":
            mse,

        "test_rmse":
            rmse,

        "test_mae":
            mae,

        "test_samples":
            len(targets_original),
    }

    # --------------------------------------------------
    # 8. Save metrics
    # --------------------------------------------------

    METRICS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        METRICS_PATH,
        "w",
    ) as f:
        json.dump(
            metrics,
            f,
            indent=2,
        )

    # --------------------------------------------------
    # 9. Plot real vs predicted
    # --------------------------------------------------

    preview = 500

    plt.figure(
        figsize=(12, 5)
    )

    plt.plot(
        targets_original[:preview],
        label="Actual",
    )

    plt.plot(
        predictions_original[:preview],
        label="Predicted",
    )

    plt.xlabel(
        "Test Sample"
    )

    plt.ylabel(
        "Original Value"
    )

    plt.title(
        "LSTM Forecast: Actual vs Predicted"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        PLOT_PATH,
        dpi=150,
    )

    # --------------------------------------------------
    # 10. Report
    # --------------------------------------------------

    print()
    print("=== TEST RESULTS ===")

    print(
        f"Samples: {len(targets_original)}"
    )

    print(
        f"MSE:  {mse:.6f}"
    )

    print(
        f"RMSE: {rmse:.6f}"
    )

    print(
        f"MAE:  {mae:.6f}"
    )

    print()
    print(
        "Metrics saved to:",
        METRICS_PATH,
    )

    print(
        "Prediction plot:",
        PLOT_PATH,
    )


if __name__ == "__main__":
    main()
