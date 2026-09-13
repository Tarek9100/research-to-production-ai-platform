import math

import mlflow
import mlflow.pytorch
import pandas as pd
import torch


MODEL_NAME = "timeseries-lstm-forecaster"
CANDIDATE_ALIAS = "candidate"

MODEL_URI = (
    f"models:/{MODEL_NAME}"
    f"@{CANDIDATE_ALIAS}"
)

DATA_PATH = "data/raw/timeseries.csv"

SEQUENCE_LENGTH = 24
BATCH_SIZE = 512


def metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
):
    error = prediction - target

    mse = torch.mean(
        error ** 2
    ).item()

    rmse = math.sqrt(mse)

    mae = torch.mean(
        torch.abs(error)
    ).item()

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
    }


def evaluate_model(
    model,
    x: torch.Tensor,
):
    predictions = []

    with torch.no_grad():
        for start in range(
            0,
            len(x),
            BATCH_SIZE,
        ):
            batch = x[
                start:
                start + BATCH_SIZE
            ]

            prediction = model(
                batch
            )

            predictions.append(
                prediction.cpu()
            )

    return torch.cat(
        predictions
    )


def main():
    print("Tracking URI:")
    print(
        mlflow.get_tracking_uri()
    )

    print()
    print("Candidate URI:")
    print(MODEL_URI)

    # --------------------------------------------------
    # 1. Load actual registered candidate artifact
    # --------------------------------------------------

    model = mlflow.pytorch.load_model(
        MODEL_URI
    )

    print()
    print(
        "Loaded candidate type:",
        type(model),
    )

    # PT2 exported model:
    # do not call model.eval().

    # --------------------------------------------------
    # 2. Load RAW chronological test split
    # --------------------------------------------------

    df = pd.read_csv(
        DATA_PATH
    )

    values = torch.tensor(
        df["value"].to_numpy(),
        dtype=torch.float32,
    )

    n = len(values)

    test_start = int(
        n * 0.85
    )

    raw_test = values[
        test_start:
    ]

    # --------------------------------------------------
    # 3. Build all test windows
    #
    # Input:
    #   [t-24 ... t-1]
    #
    # Target:
    #   t
    #
    # With a 3000-point test split and sequence=24,
    # this produces 2976 evaluation examples.
    # --------------------------------------------------

    x = []
    y = []

    for i in range(
        len(raw_test)
        - SEQUENCE_LENGTH
    ):
        sequence = raw_test[
            i:
            i + SEQUENCE_LENGTH
        ]

        target = raw_test[
            i + SEQUENCE_LENGTH
        ]

        x.append(
            sequence.unsqueeze(-1)
        )

        y.append(
            target
        )

    x = torch.stack(x)

    y = torch.stack(y)

    print()
    print(
        "Evaluation samples:",
        len(y),
    )

    print(
        "Input shape:",
        tuple(x.shape),
    )

    print(
        "Target shape:",
        tuple(y.shape),
    )

    # --------------------------------------------------
    # 4. Actual registered model
    # --------------------------------------------------

    candidate_prediction = (
        evaluate_model(
            model,
            x,
        )
    )

    candidate_metrics = metrics(
        candidate_prediction,
        y,
    )

    # --------------------------------------------------
    # 5. Persistence baseline
    #
    # Predict:
    #   y[t] = y[t-1]
    #
    # Last observation in each input window.
    # --------------------------------------------------

    persistence_prediction = (
        x[:, -1, 0]
    )

    persistence_metrics = metrics(
        persistence_prediction,
        y,
    )

    # --------------------------------------------------
    # 6. Daily seasonal-naive baseline
    #
    # Sequence length = daily period = 24.
    #
    # Predict:
    #   y[t] = y[t-24]
    #
    # This is the first observation in the
    # 24-point input window.
    # --------------------------------------------------

    daily_prediction = (
        x[:, 0, 0]
    )

    daily_metrics = metrics(
        daily_prediction,
        y,
    )

    # --------------------------------------------------
    # 7. Relative RMSE improvement
    # --------------------------------------------------

    persistence_improvement = (
        (
            persistence_metrics["rmse"]
            - candidate_metrics["rmse"]
        )
        / persistence_metrics["rmse"]
        * 100
    )

    daily_improvement = (
        (
            daily_metrics["rmse"]
            - candidate_metrics["rmse"]
        )
        / daily_metrics["rmse"]
        * 100
    )

    # --------------------------------------------------
    # 8. Report only.
    #
    # IMPORTANT:
    # no registry mutation happens here.
    # --------------------------------------------------

    print()
    print(
        "===== FULL TEST EVALUATION ====="
    )

    print()
    print("Candidate:")
    print(
        "  MSE: ",
        f'{candidate_metrics["mse"]:.6f}',
    )
    print(
        "  RMSE:",
        f'{candidate_metrics["rmse"]:.6f}',
    )
    print(
        "  MAE: ",
        f'{candidate_metrics["mae"]:.6f}',
    )

    print()
    print("Persistence baseline:")
    print(
        "  MSE: ",
        f'{persistence_metrics["mse"]:.6f}',
    )
    print(
        "  RMSE:",
        f'{persistence_metrics["rmse"]:.6f}',
    )
    print(
        "  MAE: ",
        f'{persistence_metrics["mae"]:.6f}',
    )

    print()
    print(
        "Daily seasonal-naive baseline:"
    )
    print(
        "  MSE: ",
        f'{daily_metrics["mse"]:.6f}',
    )
    print(
        "  RMSE:",
        f'{daily_metrics["rmse"]:.6f}',
    )
    print(
        "  MAE: ",
        f'{daily_metrics["mae"]:.6f}',
    )

    print()
    print(
        "Candidate RMSE improvement "
        "vs persistence:"
    )
    print(
        f"  {persistence_improvement:.2f}%"
    )

    print()
    print(
        "Candidate RMSE improvement "
        "vs daily seasonal naive:"
    )
    print(
        f"  {daily_improvement:.2f}%"
    )

    print()

    if (
        candidate_metrics["rmse"]
        < persistence_metrics["rmse"]
    ):
        print(
            "PASS: candidate beats "
            "persistence baseline."
        )
    else:
        print(
            "FAIL: candidate does NOT beat "
            "persistence baseline."
        )

    if (
        candidate_metrics["rmse"]
        < daily_metrics["rmse"]
    ):
        print(
            "PASS: candidate beats "
            "daily seasonal-naive baseline."
        )
    else:
        print(
            "FAIL: candidate does NOT beat "
            "daily seasonal-naive baseline."
        )


if __name__ == "__main__":
    main()
