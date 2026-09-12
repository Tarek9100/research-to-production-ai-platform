import mlflow
import mlflow.pytorch
import torch

from src.data.dataset import TimeSeriesDataset
from src.data.prepare import load_and_split
from src.models.lstm import LSTMForecaster


DATA_PATH = "data/raw/timeseries.csv"

CHECKPOINT_PATH = (
    "artifacts/checkpoints/lstm_mlflow_best.pt"
)

MODEL_URI = (
    "models:/m-8d3a73abc5324745aaa992fbdf641105"
)


def main():

    # --------------------------------------------------
    # 1. Load original training checkpoint
    # --------------------------------------------------

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
    )

    eager_model = LSTMForecaster(
        input_size=1,
        hidden_size=checkpoint["hidden_size"],
        num_layers=checkpoint["num_layers"],
        dropout=0.1,
    )

    eager_model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    eager_model.eval()

    # --------------------------------------------------
    # 2. Load MLflow-packaged PT2 model
    # --------------------------------------------------

    exported_model = mlflow.pytorch.load_model(
        MODEL_URI
    )

    # Do NOT call exported_model.eval().
    # PT2 model was already exported for inference.

    # --------------------------------------------------
    # 3. Recreate test data exactly as training did
    # --------------------------------------------------

    splits = load_and_split(
        DATA_PATH
    )

    test_dataset = TimeSeriesDataset(
        splits.test,
        sequence_length=
            checkpoint["sequence_length"],
    )

    # Take several real samples rather than zeros.
    samples = []

    for i in range(8):
        x, _ = test_dataset[i]
        samples.append(x)

    x = torch.stack(samples)

    print("Input shape:")
    print(x.shape)

    print()
    print("First sequence preview:")
    print(x[0, :5, 0])

    # --------------------------------------------------
    # 4. Run both model representations
    # --------------------------------------------------

    with torch.no_grad():

        eager_prediction = (
            eager_model(x)
        )

        exported_prediction = (
            exported_model(x)
        )

    # --------------------------------------------------
    # 5. Compare numerically
    # --------------------------------------------------

    absolute_difference = torch.abs(
        eager_prediction
        - exported_prediction
    )

    max_absolute_difference = (
        absolute_difference.max().item()
    )

    mean_absolute_difference = (
        absolute_difference.mean().item()
    )

    print()
    print("Checkpoint predictions:")
    print(eager_prediction)

    print()
    print("MLflow PT2 predictions:")
    print(exported_prediction)

    print()
    print(
        "Maximum absolute difference:",
        max_absolute_difference,
    )

    print(
        "Mean absolute difference:",
        mean_absolute_difference,
    )

    # Strong automated assertion.
    torch.testing.assert_close(
        eager_prediction,
        exported_prediction,
        rtol=1e-5,
        atol=1e-6,
    )

    print()
    print(
        "PASS: checkpoint and MLflow model "
        "produce equivalent predictions."
    )


if __name__ == "__main__":
    main()
