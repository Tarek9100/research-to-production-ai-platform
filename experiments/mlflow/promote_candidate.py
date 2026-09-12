import sys

import mlflow
import mlflow.pytorch
import torch

from mlflow import MlflowClient


MODEL_NAME = "timeseries-lstm-forecaster"

CANDIDATE_ALIAS = "candidate"
CHAMPION_ALIAS = "champion"

MAX_TEST_RMSE = 0.90
MAX_TEST_MAE = 0.75


def fail(message: str):
    print()
    print("PROMOTION FAILED:")
    print(message)
    sys.exit(1)


def main():

    client = MlflowClient()

    print("Tracking URI:")
    print(mlflow.get_tracking_uri())
    print()

    # --------------------------------------------------
    # 1. Resolve candidate
    # --------------------------------------------------

    candidate = (
        client.get_model_version_by_alias(
            MODEL_NAME,
            CANDIDATE_ALIAS,
        )
    )

    version = candidate.version
    tags = candidate.tags

    print("Candidate:")
    print("  Model:", candidate.name)
    print("  Version:", version)
    print("  Tags:", tags)
    print()

    # --------------------------------------------------
    # 2. Required metadata checks
    # --------------------------------------------------

    if (
        tags.get("validation_status")
        != "passed"
    ):
        fail(
            "validation_status is not passed"
        )

    if (
        tags.get("contract_equivalence")
        != "passed"
    ):
        fail(
            "contract_equivalence is not passed"
        )

    if (
        tags.get("serving_contract")
        != "raw-input-raw-output"
    ):
        fail(
            "serving contract is not approved"
        )

    # --------------------------------------------------
    # 3. Quality metric checks
    # --------------------------------------------------

    try:
        test_rmse = float(
            tags["test_rmse"]
        )

        test_mae = float(
            tags["test_mae"]
        )

    except (KeyError, ValueError):
        fail(
            "test metrics are missing or invalid"
        )

    print("Quality gate:")
    print(
        f"  RMSE: {test_rmse:.6f} "
        f"<= {MAX_TEST_RMSE:.2f}"
    )

    print(
        f"  MAE:  {test_mae:.6f} "
        f"<= {MAX_TEST_MAE:.2f}"
    )

    if test_rmse > MAX_TEST_RMSE:
        fail(
            "RMSE exceeds promotion threshold"
        )

    if test_mae > MAX_TEST_MAE:
        fail(
            "MAE exceeds promotion threshold"
        )

    # --------------------------------------------------
    # 4. Runtime smoke test
    # --------------------------------------------------

    candidate_uri = (
        f"models:/{MODEL_NAME}"
        f"@{CANDIDATE_ALIAS}"
    )

    print()
    print("Loading candidate:")
    print(candidate_uri)

    model = mlflow.pytorch.load_model(
        candidate_uri
    )

    x = torch.tensor(
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
            ]
        ],
        dtype=torch.float32,
    )

    with torch.no_grad():
        prediction = model(x)

    if prediction.shape != torch.Size([1]):
        fail(
            "unexpected model output shape"
        )

    if not torch.isfinite(
        prediction
    ).all():
        fail(
            "model produced non-finite output"
        )

    print(
        "Smoke-test prediction:",
        prediction,
    )

    # --------------------------------------------------
    # 5. Promote candidate -> champion
    # --------------------------------------------------

    client.set_registered_model_alias(
        MODEL_NAME,
        CHAMPION_ALIAS,
        version,
    )

    client.set_model_version_tag(
        MODEL_NAME,
        version,
        "promotion_status",
        "champion",
    )

    client.set_model_version_tag(
        MODEL_NAME,
        version,
        "quality_gate",
        "passed",
    )

    # --------------------------------------------------
    # 6. Verify
    # --------------------------------------------------

    champion = (
        client.get_model_version_by_alias(
            MODEL_NAME,
            CHAMPION_ALIAS,
        )
    )

    print()
    print("PROMOTION PASSED")

    print(
        "champion -> version",
        champion.version,
    )

    print()
    print("Champion URI:")

    print(
        f"models:/{MODEL_NAME}"
        f"@{CHAMPION_ALIAS}"
    )


if __name__ == "__main__":
    main()
