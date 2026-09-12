from pathlib import Path
import random
import time

import mlflow
import mlflow.pytorch
from mlflow.models import ModelSignature
from mlflow.types.schema import Schema, TensorSpec
import matplotlib.pyplot as plt
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

EXPERIMENT_NAME = (
    "research-to-production-ai-platform"
)

CHECKPOINT_PATH = Path(
    "artifacts/checkpoints/lstm_mlflow_best.pt"
)

EVALUATION_DIR = Path(
    "artifacts/evaluation/mlflow"
)

TEST_METRICS_PATH = (
    EVALUATION_DIR / "test_metrics.json"
)

TEST_PLOT_PATH = (
    EVALUATION_DIR / "test_predictions.png"
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

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=pin_memory,
    )


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



def evaluate_test_set(
    checkpoint_path,
    test_values,
    device,
):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model = LSTMForecaster(
        input_size=1,
        hidden_size=checkpoint["hidden_size"],
        num_layers=checkpoint["num_layers"],
        dropout=DROPOUT,
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    test_dataset = TimeSeriesDataset(
        test_values,
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

    mean = checkpoint["mean"]
    std = checkpoint["std"]

    predictions_original = (
        predictions * std + mean
    )

    targets_original = (
        targets * std + mean
    )

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
        np.mean(np.abs(errors))
    )

    EVALUATION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics = {
        "test_mse": mse,
        "test_rmse": rmse,
        "test_mae": mae,
        "test_samples":
            len(targets_original),
        "checkpoint_epoch":
            checkpoint["epoch"],
    }

    import json

    with open(
        TEST_METRICS_PATH,
        "w",
    ) as f:
        json.dump(
            metrics,
            f,
            indent=2,
        )

    preview = min(
        500,
        len(targets_original),
    )

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

    plt.xlabel("Test sample")
    plt.ylabel("Value")

    plt.title(
        "LSTM Test Predictions"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        TEST_PLOT_PATH,
        dpi=150,
    )

    plt.close()

    return metrics, model


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

    splits = load_and_split(
        DATA_PATH
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

    CHECKPOINT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    with mlflow.start_run(
        run_name="lstm-baseline-gpu"
    ) as run:

        print("MLflow run ID:")
        print(run.info.run_id)
        print()

        mlflow.log_params(
            {
                "model":
                    "LSTMForecaster",

                "sequence_length":
                    SEQUENCE_LENGTH,

                "batch_size":
                    BATCH_SIZE,

                "hidden_size":
                    HIDDEN_SIZE,

                "num_layers":
                    NUM_LAYERS,

                "dropout":
                    DROPOUT,

                "learning_rate":
                    LEARNING_RATE,

                "epochs":
                    EPOCHS,

                "seed":
                    SEED,

                "device":
                    device.type,

                "parameter_count":
                    parameter_count,

                "train_observations":
                    len(splits.train),

                "validation_observations":
                    len(splits.validation),

                "test_observations":
                    len(splits.test),
            }
        )

        mlflow.set_tag(
            "project",
            "research-to-production-ai-platform"
        )

        mlflow.set_tag(
            "workload",
            "time-series-forecasting"
        )

        if device.type == "cuda":

            mlflow.log_param(
                "gpu_name",
                torch.cuda.get_device_name(0),
            )

            mlflow.log_param(
                "pytorch_cuda_runtime",
                torch.version.cuda,
            )

            torch.cuda.reset_peak_memory_stats()

        mlflow.log_param(
            "pytorch_version",
            torch.__version__,
        )

        print("Device:", device)

        if device.type == "cuda":
            print(
                "GPU:",
                torch.cuda.get_device_name(0),
            )

        print(
            "Model parameters:",
            parameter_count,
        )

        print()

        best_validation_loss = float("inf")

        training_start = (
            time.perf_counter()
        )

        total_samples_seen = 0
        pure_training_time = 0.0

        for epoch in range(
            1,
            EPOCHS + 1,
        ):

            epoch_start = (
                time.perf_counter()
            )

            model.train()

            running_loss = 0.0
            samples_seen = 0

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

                samples_seen += (
                    batch_size
                )

            if device.type == "cuda":
                torch.cuda.synchronize()

            epoch_time = (
                time.perf_counter()
                - epoch_start
            )

            pure_training_time += epoch_time

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

            epoch_throughput = (
                samples_seen
                / epoch_time
            )

            total_samples_seen += (
                samples_seen
            )

            mlflow.log_metric(
                "train_loss",
                train_loss,
                step=epoch,
            )

            mlflow.log_metric(
                "validation_loss",
                validation_loss,
                step=epoch,
            )

            mlflow.log_metric(
                "epoch_time_seconds",
                epoch_time,
                step=epoch,
            )

            mlflow.log_metric(
                "epoch_throughput_samples_sec",
                epoch_throughput,
                step=epoch,
            )

            print(
                f"Epoch {epoch:02d}/{EPOCHS} "
                f"train={train_loss:.6f} "
                f"val={validation_loss:.6f} "
                f"time={epoch_time:.3f}s "
                f"throughput="
                f"{epoch_throughput:.1f} samples/s"
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

        if device.type == "cuda":
            torch.cuda.synchronize()

        total_training_time = (
            time.perf_counter()
            - training_start
        )

        pure_training_throughput = (
            total_samples_seen
            / pure_training_time
        )

        end_to_end_throughput = (
            total_samples_seen
            / total_training_time
        )

        mlflow.log_metric(
            "best_validation_loss",
            best_validation_loss,
        )

        mlflow.log_metric(
            "total_training_time_seconds",
            total_training_time,
        )

        mlflow.log_metric(
            "pure_training_throughput_samples_sec",
            pure_training_throughput,
        )

        mlflow.log_metric(
            "end_to_end_throughput_samples_sec",
            end_to_end_throughput,
        )

        if device.type == "cuda":

            peak_gpu_memory = (
                torch.cuda.max_memory_allocated()
                / 1024**2
            )

            mlflow.log_metric(
                "peak_gpu_memory_mib",
                peak_gpu_memory,
            )

        mlflow.log_artifact(
            str(CHECKPOINT_PATH),
            artifact_path="checkpoints",
        )

        test_metrics, best_model = (
            evaluate_test_set(
                checkpoint_path=
                    CHECKPOINT_PATH,
                test_values=
                    splits.test,
                device=device,
            )
        )

        mlflow.log_metrics(
            {
                "test_mse":
                    test_metrics["test_mse"],

                "test_rmse":
                    test_metrics["test_rmse"],

                "test_mae":
                    test_metrics["test_mae"],
            }
        )

        mlflow.log_param(
            "test_samples",
            test_metrics["test_samples"],
        )

        mlflow.log_param(
            "selected_checkpoint_epoch",
            test_metrics[
                "checkpoint_epoch"
            ],
        )

        mlflow.log_artifact(
            str(TEST_METRICS_PATH),
            artifact_path="evaluation",
        )

        mlflow.log_artifact(
            str(TEST_PLOT_PATH),
            artifact_path="evaluation",
        )

        # Package the selected model on CPU.
        #
        # Training used CUDA, but the MLflow model artifact should
        # not be tied to the GPU that happened to produce it.
        best_model = best_model.to("cpu")
        best_model.eval()

        # Use batch size 2 for the export example.
        #
        # Shape:
        #   [batch, sequence_length, features]
        #
        #   [2, 24, 1]
        input_example = torch.zeros(
            (
                2,
                SEQUENCE_LENGTH,
                1,
            ),
            dtype=torch.float32,
        )

        with torch.no_grad():
            output_example = (
                best_model(input_example)
            )

        # Explicit tensor contract:
        #
        # input:
        #   variable batch x 24 timesteps x 1 feature
        #
        # output:
        #   variable batch of scalar predictions
        signature = ModelSignature(
            inputs=Schema(
                [
                    TensorSpec(
                        np.dtype(np.float32),
                        (-1, SEQUENCE_LENGTH, 1),
                    )
                ]
            ),
            outputs=Schema(
                [
                    TensorSpec(
                        np.dtype(np.float32),
                        (-1,),
                    )
                ]
            ),
        )

        model_info = (
            mlflow.pytorch.log_model(
                best_model,
                name="model",
                input_example=input_example,
                signature=signature,
                code_paths=["src"],
                serialization_format="pt2",
            )
        )

        print(
            "Logged MLflow model:",
            model_info.model_uri,
        )

        print()
        print("=== TEST RESULTS ===")

        print(
            "MSE:",
            f'{test_metrics["test_mse"]:.6f}',
        )

        print(
            "RMSE:",
            f'{test_metrics["test_rmse"]:.6f}',
        )

        print(
            "MAE:",
            f'{test_metrics["test_mae"]:.6f}',
        )

        print()
        print(
            "Best validation loss:",
            f"{best_validation_loss:.6f}",
        )

        print(
            "Total training time:",
            f"{total_training_time:.3f}s",
        )

        print(
            "Pure training throughput:",
            f"{pure_training_throughput:.1f} samples/s",
        )

        print(
            "End-to-end throughput:",
            f"{end_to_end_throughput:.1f} samples/s",
        )

        print(
            "Checkpoint:",
            CHECKPOINT_PATH,
        )


if __name__ == "__main__":
    main()
