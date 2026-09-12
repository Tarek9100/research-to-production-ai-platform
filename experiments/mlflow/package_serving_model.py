import mlflow
import mlflow.pytorch
import numpy as np
import torch

from mlflow.models import ModelSignature
from mlflow.types.schema import Schema, TensorSpec

from src.models.lstm import LSTMForecaster
from src.models.serving import TimeSeriesServingModel


CHECKPOINT_PATH = (
    "artifacts/checkpoints/lstm_mlflow_best.pt"
)

SOURCE_RUN_ID = (
    "d209f41a5a2b4dd7ac5a61e3cb74c8c3"
)

SOURCE_MODEL_ID = (
    "m-8d3a73abc5324745aaa992fbdf641105"
)

EXPERIMENT_NAME = (
    "research-to-production-ai-platform"
)


def main():

    # --------------------------------------------------
    # 1. Load selected checkpoint
    # --------------------------------------------------

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
    )

    base_model = LSTMForecaster(
        input_size=1,
        hidden_size=checkpoint[
            "hidden_size"
        ],
        num_layers=checkpoint[
            "num_layers"
        ],
        dropout=0.1,
    )

    base_model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    base_model.eval()

    # --------------------------------------------------
    # 2. Wrap preprocessing + model + postprocessing
    # --------------------------------------------------

    serving_model = (
        TimeSeriesServingModel(
            model=base_model,
            mean=checkpoint["mean"],
            std=checkpoint["std"],
        )
    )

    serving_model.eval()

    print("Training mean:")
    print(checkpoint["mean"])

    print("Training std:")
    print(checkpoint["std"])

    print()

    # --------------------------------------------------
    # 3. Test using RAW values
    # --------------------------------------------------

    raw_example = torch.tensor(
        [
            [
                [18.0],
                [18.2],
                [18.1],
                [17.9],
                [18.3],
                [18.5],
                [18.4],
                [18.6],
                [18.7],
                [18.5],
                [18.8],
                [19.0],
                [18.9],
                [19.1],
                [19.2],
                [19.0],
                [19.3],
                [19.4],
                [19.2],
                [19.5],
                [19.6],
                [19.4],
                [19.7],
                [19.8],
            ],
        ],
        dtype=torch.float32,
    )

    with torch.no_grad():
        raw_prediction = (
            serving_model(raw_example)
        )

    print("Raw input shape:")
    print(raw_example.shape)

    print(
        "Raw-scale prediction:",
        raw_prediction,
    )

    # --------------------------------------------------
    # 4. Define serving contract
    # --------------------------------------------------

    signature = ModelSignature(
        inputs=Schema(
            [
                TensorSpec(
                    np.dtype(np.float32),
                    (-1, 24, 1),
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

    input_example = torch.tensor(
        np.repeat(
            raw_example.numpy(),
            repeats=2,
            axis=0,
        ),
        dtype=torch.float32,
    )

    # --------------------------------------------------
    # 5. Log deployable model
    # --------------------------------------------------

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    with mlflow.start_run(
        run_name="lstm-serving-package"
    ) as run:

        mlflow.set_tag(
            "model_role",
            "serving",
        )

        mlflow.set_tag(
            "source_training_run_id",
            SOURCE_RUN_ID,
        )

        mlflow.set_tag(
            "source_logged_model_id",
            SOURCE_MODEL_ID,
        )

        mlflow.log_param(
            "sequence_length",
            checkpoint["sequence_length"],
        )

        mlflow.log_param(
            "normalization_mean",
            checkpoint["mean"],
        )

        mlflow.log_param(
            "normalization_std",
            checkpoint["std"],
        )

        mlflow.log_param(
            "source_checkpoint_epoch",
            checkpoint["epoch"],
        )

        model_info = (
            mlflow.pytorch.log_model(
                serving_model,
                name="serving-model",
                input_example=input_example,
                signature=signature,
                code_paths=["src"],
                serialization_format="pt2",
            )
        )

        print()
        print("Packaging run ID:")
        print(run.info.run_id)

        print()
        print("Serving model URI:")
        print(model_info.model_uri)


if __name__ == "__main__":
    main()
