import mlflow
import mlflow.pytorch
import pandas as pd
import torch

from src.models.lstm import LSTMForecaster


DATA_PATH = "data/raw/timeseries.csv"

CHECKPOINT_PATH = (
    "artifacts/checkpoints/lstm_mlflow_best.pt"
)

SERVING_MODEL_URI = (
    "models:/m-c5641547ccae4f7baf0fc4c8e8f35765"
)

SEQUENCE_LENGTH = 24


def main():

    # --------------------------------------------------
    # 1. Load original checkpoint
    # --------------------------------------------------

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
    )

    base_model = LSTMForecaster(
        input_size=1,
        hidden_size=checkpoint["hidden_size"],
        num_layers=checkpoint["num_layers"],
        dropout=0.1,
    )

    base_model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    base_model.eval()

    # --------------------------------------------------
    # 2. Load packaged serving model
    # --------------------------------------------------

    serving_model = mlflow.pytorch.load_model(
        SERVING_MODEL_URI
    )

    # PT2 exported model:
    # do not call .eval()

    # --------------------------------------------------
    # 3. Load RAW data
    # --------------------------------------------------

    df = pd.read_csv(
        DATA_PATH
    )

    values = torch.tensor(
        df["value"].to_numpy(),
        dtype=torch.float32,
    )

    # Same chronological split:
    #
    # 70% train
    # 15% validation
    # 15% test
    n = len(values)

    test_start = int(
        n * 0.85
    )

    raw_test = values[
        test_start:
    ]

    # Build 8 real raw input sequences.
    samples = []

    for i in range(8):

        sequence = raw_test[
            i:
            i + SEQUENCE_LENGTH
        ]

        sequence = (
            sequence.unsqueeze(-1)
        )

        samples.append(
            sequence
        )

    raw_x = torch.stack(
        samples
    )

    print("Raw input shape:")
    print(raw_x.shape)

    print()
    print(
        "First raw sequence preview:"
    )

    print(
        raw_x[0, :5, 0]
    )

    # --------------------------------------------------
    # 4. Path A:
    #    manual preprocessing + checkpoint
    # --------------------------------------------------

    mean = checkpoint["mean"]
    std = checkpoint["std"]

    normalized_x = (
        raw_x - mean
    ) / std

    with torch.no_grad():

        normalized_prediction = (
            base_model(
                normalized_x
            )
        )

        manual_prediction = (
            normalized_prediction
            * std
            + mean
        )

    # --------------------------------------------------
    # 5. Path B:
    #    self-contained MLflow serving model
    # --------------------------------------------------

    with torch.no_grad():

        serving_prediction = (
            serving_model(
                raw_x
            )
        )

    # --------------------------------------------------
    # 6. Compare
    # --------------------------------------------------

    difference = torch.abs(
        manual_prediction
        - serving_prediction
    )

    max_difference = (
        difference.max().item()
    )

    mean_difference = (
        difference.mean().item()
    )

    print()
    print(
        "Manual pipeline predictions:"
    )

    print(
        manual_prediction
    )

    print()
    print(
        "Serving model predictions:"
    )

    print(
        serving_prediction
    )

    print()
    print(
        "Maximum absolute difference:",
        max_difference,
    )

    print(
        "Mean absolute difference:",
        mean_difference,
    )

    torch.testing.assert_close(
        manual_prediction,
        serving_prediction,
        rtol=1e-5,
        atol=1e-6,
    )

    print()
    print(
        "PASS: self-contained serving "
        "model matches the original "
        "preprocessing + checkpoint + "
        "postprocessing pipeline."
    )


if __name__ == "__main__":
    main()
